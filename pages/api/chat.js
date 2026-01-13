import { Pinecone } from '@pinecone-database/pinecone';
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
    return res.status(405).end();
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
    // RAG
    const queryEmbeddingResponse = await hf.featureExtraction({
      model: 'intfloat/multilingual-e5-large',
      inputs: `query: ${latestUserMessage}`,
    });

    const queryEmbedding = Array.from(queryEmbeddingResponse);

    const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

    const queryResponse = await index.query({
      vector: queryEmbedding,
      topK: 5,
      includeMetadata: true,
    });

    const matches = queryResponse.matches || [];

    let context = '';
    if (matches.length > 0) {
      const relevantMatches = matches.filter(m => m.score > 0.4).slice(0, 5);
      context = relevantMatches
        .map(match => match.metadata?.text || '')
        .filter(text => text.trim() !== '')
        .join('\n\n');
    }

    const systemPrompt = {
      role: 'system',
      content: `Du är en hjälpsam, vänlig och artig supportagent för FortusPay.
Svara ALLTID på EXAKT samma språk som kundens senaste fråga – högsta prioritet.
Översätt HELA kunskapsbasen till kundens språk. Behåll struktur, numrering och detaljer.
Avsluta med "Behöver du hjälp med något mer?" på kundens språk.

Om svaret kan variera beroende på produkt/terminal, fråga efter förtydligande.

Använd ENDAST kunskapsbasen.
Avsluta varje svar med: "Detta är ett AI-genererat svar. För bindande råd, kontakta support@fortuspay.se." (eller översätt till kundens språk).

Kunskapsbas (översätt till kundens språk):
${context}`
    };

    const groqMessages = [systemPrompt, ...messages];

    // Streaming
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const stream = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: groqMessages,
      temperature: 0.3,
      max_tokens: 1024,
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      if (content) {
        res.write(`data: ${JSON.stringify({ content })}\n\n`);
      }
    }

    res.write('data: [DONE]\n\n');
    res.end();
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}

export const config = {
  api: {
    bodyParser: true,
  },
};
