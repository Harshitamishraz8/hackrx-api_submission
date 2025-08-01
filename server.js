const express = require('express');
const cors = require('cors');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const { Pinecone } = require('@pinecone-database/pinecone');
const OpenAI = require('openai');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize clients
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Auth middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  
  if (!token || token !== process.env.HACKRX_API_TOKEN) {
    return res.status(401).json({ error: 'Invalid authentication token' });
  }
  
  next();
};

// Helper functions
async function downloadPDF(url) {
  try {
    const response = await axios.get(url, {
      responseType: 'arraybuffer',
      timeout: 30000
    });
    return Buffer.from(response.data);
  } catch (error) {
    console.error('Error downloading PDF:', error.message);
    throw new Error('Failed to download PDF');
  }
}

async function extractTextFromPDF(pdfBuffer) {
  try {
    const data = await pdfParse(pdfBuffer);
    return data.text;
  } catch (error) {
    console.error('Error extracting text from PDF:', error.message);
    throw new Error('Failed to extract text from PDF');
  }
}

function chunkText(text, chunkSize = 1000, overlap = 200) {
  const chunks = [];
  const words = text.split(' ');
  
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunk = words.slice(i, i + chunkSize).join(' ');
    if (chunk.trim().length > 0) {
      chunks.push(chunk.trim());
    }
  }
  
  return chunks;
}

async function getEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: text,
    });
    return response.data[0].embedding;
  } catch (error) {
    console.error('Error getting embedding:', error.message);
    throw new Error('Failed to get embedding');
  }
}

async function storeInPinecone(chunks, docId) {
  try {
    const index = pinecone.index(process.env.PINECONE_INDEX);
    
    const vectors = [];
    for (let i = 0; i < chunks.length; i++) {
      const embedding = await getEmbedding(chunks[i]);
      vectors.push({
        id: `${docId}_chunk_${i}`,
        values: embedding,
        metadata: {
          text: chunks[i],
          docId: docId,
          chunkIndex: i
        }
      });
    }
    
    await index.upsert(vectors);
    return true;
  } catch (error) {
    console.error('Error storing in Pinecone:', error.message);
    throw new Error('Failed to store in Pinecone');
  }
}

async function queryPinecone(question, docId, topK = 5) {
  try {
    const index = pinecone.index(process.env.PINECONE_INDEX);
    const questionEmbedding = await getEmbedding(question);
    
    const queryResponse = await index.query({
      vector: questionEmbedding,
      topK: topK,
      filter: { docId: docId },
      includeMetadata: true
    });
    
    return queryResponse.matches.map(match => match.metadata.text);
  } catch (error) {
    console.error('Error querying Pinecone:', error.message);
    return [];
  }
}

async function generateAnswer(question, contexts) {
  try {
    const contextText = contexts.join('\n\n');
    const prompt = `Based on the following context, answer the question. If the answer cannot be found in the context, say "Information not available in the provided documents."

Context:
${contextText}

Question: ${question}

Answer:`;

    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "user",
          content: prompt
        }
      ],
      max_tokens: 500,
      temperature: 0.2
    });

    return response.choices[0].message.content.trim();
  } catch (error) {
    console.error('Error generating answer:', error.message);
    return "Error generating answer.";
  }
}

// Routes
app.get('/', (req, res) => {
  res.json({
    status: "API is running",
    message: "HackRx Document Q&A API",
    endpoint: "/hackrx/run"
  });
});

app.post('/hackrx/run', authenticateToken, async (req, res) => {
  try {
    const { documents, questions } = req.body;
    
    if (!documents || !questions || !Array.isArray(questions)) {
      return res.status(400).json({
        error: "Invalid request format. Expected 'documents' (string) and 'questions' (array)."
      });
    }

    console.log(`Processing ${questions.length} questions for document: ${documents}`);
    
    // Generate document ID
    const docId = Buffer.from(documents).toString('base64').slice(0, 8);
    
    // Download and process PDF
    const pdfBuffer = await downloadPDF(documents);
    const text = await extractTextFromPDF(pdfBuffer);
    
    if (!text || text.trim().length === 0) {
      return res.status(400).json({
        error: "No text could be extracted from the PDF"
      });
    }
    
    // Chunk text and store in Pinecone
    const chunks = chunkText(text);
    await storeInPinecone(chunks, docId);
    
    // Generate answers
    const answers = [];
    for (const question of questions) {
      const contexts = await queryPinecone(question, docId, 5);
      const answer = await generateAnswer(question, contexts);
      answers.push(answer);
    }
    
    console.log(`Generated ${answers.length} answers`);
    
    res.json({ answers });
    
  } catch (error) {
    console.error('Error in /hackrx/run:', error.message);
    res.status(500).json({
      error: `Internal server error: ${error.message}`
    });
  }
});

app.listen(port, () => {
  console.log(`HackRx API server running on port ${port}`);
});