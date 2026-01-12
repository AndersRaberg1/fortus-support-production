import { HfInference } from '@huggingface/inference';
import { Pinecone } from '@pinecone-database/pinecone';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { message } = req.body || {};

  try {
    const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME || 'fortus-support-hf');

    // Embed message med HF
    const embeddingResponse = await hf.featureExtraction({
      model: 'sentence-transformers/all-MiniLM-L6-v2',
      inputs: message,
    });
    const queryEmbedding = Array.from(embeddingResponse);

    // Query Pinecone
    const queryResponse = await index.query({
      vector: queryEmbedding,
      topK: 5,
      includeMetadata: true,
    });

    const context = queryResponse.matches
      .map(m => m.metadata.text || '')
      .join('\n\n') || 'Ingen relevant kunskap hittades.';

    // Prompt
    const prompt = `Du är en hjälpsam support-AI för FortusPay. Använd ENDAST denna kunskap för svaret (inga påhitt): ${context}\n\nFråga: ${message}\nSvara på svenska, kort och stegvis.`;

    // Groq
    const groqResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'llama-3.1-8b-instant',
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    if (!groqResponse.ok) {
      throw new Error('Groq API error');
    }

    const data = await groqResponse.json();
    const reply = data.choices[0]?.message?.content || 'Inget svar från AI.';

    res.status(200).json({ response: reply });
  } catch (error) {
    console.error('RAG error:', error);
    res.status(500).json({ error: 'Fel vid RAG: ' + (error.message || 'Okänt fel') });
  }
}
