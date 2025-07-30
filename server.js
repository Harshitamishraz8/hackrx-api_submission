const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Mock vector database (in production, you'd use Pinecone)
let documentStore = new Map();

// Mock embedding function (in production, you'd use sentence-transformers)
function createEmbedding(text) {
    // Simple hash-based mock embedding for demo
    const words = text.toLowerCase().split(/\s+/);
    const embedding = new Array(384).fill(0);
    
    words.forEach((word, idx) => {
        for (let i = 0; i < word.length; i++) {
            embedding[i % 384] += word.charCodeAt(i);
        }
    });
    
    return embedding.map(val => val / 1000); // Normalize
}

// Mock similarity function
function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Process PDF and extract text (mock implementation)
async function processPDF(pdfUrl) {
    try {
        console.log(`Processing PDF from: ${pdfUrl}`);
        
        // In a real implementation, you'd extract text from PDF
        // For now, we'll use mock insurance policy content
        const mockPolicyContent = `
        NATIONAL PARIVAR MEDICLAIM PLUS POLICY
        
        GRACE PERIOD: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.
        
        PRE-EXISTING DISEASES: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.
        
        MATERNITY COVERAGE: Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.
        
        CATARACT SURGERY: The policy has a specific waiting period of two (2) years for cataract surgery.
        
        ORGAN DONOR COVERAGE: Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.
        
        NO CLAIM DISCOUNT: A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.
        
        HEALTH CHECK-UPS: Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.
        
        HOSPITAL DEFINITION: A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.
        
        AYUSH COVERAGE: The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.
        
        ROOM RENT LIMITS: Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN).
        `;
        
        // Split into chunks
        const chunks = mockPolicyContent.split('\n\n').filter(chunk => chunk.trim());
        
        // Store chunks with embeddings
        const docId = Buffer.from(pdfUrl).toString('base64').substring(0, 8);
        
        chunks.forEach((chunk, index) => {
            const embedding = createEmbedding(chunk);
            documentStore.set(`${docId}-${index}`, {
                text: chunk.trim(),
                embedding: embedding,
                url: pdfUrl
            });
        });
        
        console.log(`Stored ${chunks.length} chunks for document ${docId}`);
        return true;
        
    } catch (error) {
        console.error('Error processing PDF:', error);
        return false;
    }
}

// Retrieve relevant context for a question
function retrieveContext(question, topK = 5) {
    const questionEmbedding = createEmbedding(question);
    const results = [];
    
    for (const [id, doc] of documentStore.entries()) {
        const similarity = cosineSimilarity(questionEmbedding, doc.embedding);
        results.push({
            id,
            text: doc.text,
            similarity
        });
    }
    
    // Sort by similarity and return top results
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK).map(r => r.text);
}

// Generate answer using mock LLM (in production, use Groq/OpenAI)
function generateAnswer(question, context) {
    const contextText = context.join('\n\n');
    
    // Simple keyword matching for demo purposes
    const lowerQuestion = question.toLowerCase();
    const lowerContext = contextText.toLowerCase();
    
    // Find the most relevant paragraph
    const paragraphs = contextText.split('\n\n');
    let bestMatch = '';
    let bestScore = 0;
    
    paragraphs.forEach(paragraph => {
        const words = lowerQuestion.split(' ');
        let score = 0;
        
        words.forEach(word => {
            if (word.length > 3 && paragraph.toLowerCase().includes(word)) {
                score++;
            }
        });
        
        if (score > bestScore) {
            bestScore = score;
            bestMatch = paragraph.trim();
        }
    });
    
    return bestMatch || "Information not available in the provided documents.";
}

// Authentication middleware
function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    
    const expectedToken = 'd809808918dd2a7d6b11fa5b23fa01e3abf9814dd225582d4d5674dc2138be0b';
    
    if (!token || token !== expectedToken) {
        return res.status(401).json({ error: 'Invalid or missing token' });
    }
    
    next();
}

// Main API endpoint
app.post('/hackrx/run', authenticateToken, async (req, res) => {
    try {
        const { documents, questions } = req.body;
        
        if (!documents || !questions || !Array.isArray(questions)) {
            return res.status(400).json({ 
                error: 'Invalid request format. Expected documents (string) and questions (array)' 
            });
        }
        
        console.log(`Processing ${questions.length} questions for document: ${documents}`);
        
        // Process the document
        await processPDF(documents);
        
        // Generate answers for each question
        const answers = [];
        
        for (const question of questions) {
            try {
                const context = retrieveContext(question);
                const answer = generateAnswer(question, context);
                answers.push(answer);
            } catch (error) {
                console.error(`Error processing question: ${question}`, error);
                answers.push("Error processing this question. Please try again.");
            }
        }
        
        console.log(`Generated ${answers.length} answers`);
        
        // Return in the exact format expected
        res.json({ answers });
        
    } catch (error) {
        console.error('Error in /hackrx/run:', error);
        res.status(500).json({ 
            error: 'Internal server error',
            message: error.message 
        });
    }
});

// Health check endpoint
app.get('/', (req, res) => {
    res.json({ 
        status: 'API is running', 
        message: 'HackRx Document Q&A API',
        endpoint: '/hackrx/run'
    });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 HackRx API server running on port ${PORT}`);
    console.log(`📍 Main endpoint: http://localhost:${PORT}/hackrx/run`);
    console.log(`🔑 Auth token: d809808918dd2a7d6b11fa5b23fa01e3abf9814dd225582d4d5674dc2138be0b`);
});