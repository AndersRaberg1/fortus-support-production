import { NextResponse } from 'next/server';
import Groq from 'groq-sdk';

export default async function handler(req, res) {
  if (req.method === 'POST') {
    console.log('=== API-anrop startat: POST /api/chat ===');

    try {
      const body = req.body;
      console.log('Parsad body:', JSON.stringify(body, null, 2));
      const { message } = body;

      const apiKey = process.env.GROQ_API_KEY;
      console.log('GROQ_API_KEY exists:', !!apiKey);
      if (!apiKey) throw new Error('GROQ_API_KEY saknas');

      const groq = new Groq({ apiKey });

      const completion = await groq.chat.completions.create({
        messages: [
          { role: "system", content: "Du är en hjälpsam support-AI för FortusPay. Svara vänligt på svenska." },
          { role: "user", content: message },
        ],
        model: "llama-3.1-8b-instant",  // Uppdaterad modell
      });

      const aiReply = completion.choices[0].message.content || 'Inget svar från AI – prova igen.';
      console.log('AI-svar från Groq:', aiReply);

      console.log('=== API-anrop lyckades ===');
      res.status(200).json({ reply: aiReply });
    } catch (error) {
      console.error('=== FEL I API ===');
      console.error('Error message:', error.message);
      console.error('Error stack:', error.stack);
      res.status(500).json({ error: 'Internt serverfel: ' + error.message });
    }
  } else {
    res.setHeader('Allow', ['POST']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
}
