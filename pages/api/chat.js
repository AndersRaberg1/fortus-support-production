import { Pinecone } from '@pinecone-database/pinecone';

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).end();

  const { message } = req.body;

  try {
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const index = pinecone.Index('fortus-support');

    // Embed query med samma modell
    const embedResponse = await pinecone.inference.embed('llama-text-embed-v2', [message], { input_type: 'query' });
    const queryEmbedding = embedResponse.data[0].values;

    // Query Pinecone
    const queryResponse = await index.query({
      vector: queryEmbedding,
      topK: 5,
      includeMetadata: true
    });

    const context = queryResponse.matches.map(m => m.metadata.text || '').join('\n\n');

    // Prompt med context (tvinga användning)
    const prompt = `Du är support för FortusPay. Använd ENDAST denna kunskap för svaret (inga påhitt): ${context || 'Ingen relevant kunskap hittades.'}\nFråga: ${message}\nSvara på svenska, kort och stegvis.`;

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

    const data = await groqResponse.json();
    const reply = data.choices[0].message.content;

    res.status(200).json({ response: reply });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Fel vid RAG' });
  }
}
