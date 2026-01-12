import { Pinecone } from '@pinecone-database/pinecone';
import { Groq } from 'groq-sdk';
import { HfInference } from '@huggingface/inference';

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const hf = new HfInference(process.env.HF_TOKEN);

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { messages } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'Messages array is required' });
  }

  const latestUserMessage = [...messages].reverse().find(m => m.role === 'user')?.content || '';
  
  if (!latestUserMessage.trim()) {
    return res.status(400).json({ error: 'No user message found' });
  }

  try {
    console.log('Received messages count:', messages.length);
    console.log('Latest user message for RAG:', latestUserMessage);

    const queryEmbeddingResponse = await hf.featureExtraction({
      model: 'intfloat/multilingual-e5-large',
      inputs: `query: ${latestUserMessage}`,
    });

    const queryEmbedding = Array.from(queryEmbeddingResponse);
    console.log('Query embedding length:', queryEmbedding.length);

    const indexName = process.env.PINECONE_INDEX_NAME;
    console.log('Using Pinecone index:', indexName);
    const index = pinecone.index(indexName);

    const queryResponse = await index.query({
      vector: queryEmbedding,
      topK: 5,
      includeMetadata: true,
    });

    const matches = queryResponse.matches || [];
    console.log('Number of matches found:', matches.length);

    let context = '';
    if (matches.length > 0) {
      const relevantMatches = matches.filter(m => m.score > 0.4).slice(0, 5);
      context = relevantMatches
        .map(match => match.metadata?.text || '')
        .filter(text => text.trim() !== '')
        .join('\n\n');
      console.log('Final context length:', context.length);
    } else {
      console.log('No relevant matches found');
    }

    // Starkare prompt för språk-matchning
    const systemPrompt = {
      role: 'system',
      content: `Du är en hjälpsam, vänlig och artig supportagent för FortusPay.
Svara ALLTID på exakt samma språk som kundens senaste fråga.
Översätt ALL information från kunskapsbasen till frågans språk – behåll exakt betydelse och struktur.
Var professionell men personlig – använd "du" på svenska eller "you" på engelska.
Avsluta gärna med "Behöver du hjälp med något mer?" på svenska eller "Do you need help with anything else?" på engelska.

Du FÅR INTE hitta på information. Använd ENDAST kunskapsbasen.
Om ingen relevant info: "Jag kunde tyvärr inte hitta information om detta i vår kunskapsbas. Kontakta support@fortuspay.se för hjälp." (eller exakt översättning på engelska: "I'm sorry, I couldn't find information about this in our knowledge base. Please contact support@fortuspay.se for help.")

Kunskapsbas:
${context}`
    };

    const groqMessages = [systemPrompt, ...messages];

    console.log('Sending to Groq with messages count:', groqMessages.length);

    const completion = await groq.chat.completions.create({
      model: 'llama-3.1-8b-instant',
      messages: groqMessages,
      temperature: 0.3,
      max_tokens: 1024,
      stream: false,
    });

    const answer = completion.choices[0]?.message?.content || 'Inget svar från modellen.';

    console.log('Groq response:', answer.substring(0, 500));

    res.status(200).json({ answer });
  } catch (error) {
    console.error('Error in chat API:', error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
}

export const config = {
  api: {
    bodyParser: true,
  },
};
