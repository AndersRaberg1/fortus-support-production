import { Pinecone } from '@pinecone-database/pinecone';
import { v4 as uuid } from 'uuid';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const index = pinecone.Index('fortus-support');

    // Test med bara Swish-chunk
    const chunks = [
      { keyword: 'Swish', text: 'För att använda Swish som betalmetod för din webshop eller POS, följ dessa steg: Kontakta banken. Begär om att få Swish Handel kopplat till ditt företagskonto. Ange Fortus som teknisk leverantör när du pratar med banken. Fortus tekniska leverantörs ID är 9873196894. Uppge detta ID för att slutföra kopplingen. Automatisk aktivering. Efter att Swish är kopplat med Fortus som teknisk leverantör, kommer Fortus att automatiskt aktivera Swish i de valda försäljningskanalerna. När detta är klart kan du börja ta emot Swish-betalningar från dina kunder i de kanaler du har valt.' },
    ];

    const vectors = [];
    for (const chunk of chunks) {
      const embedResponse = await pinecone.inference.embed(
        'llama-text-embed-v2',
        [chunk.text],
        { input_type: 'passage' }  // Detta löser felet!
      );
      const embedding = embedResponse.data[0].values;
      vectors.push({
        id: uuid(),
        values: embedding,
        metadata: { keyword: chunk.keyword, text: chunk.text }
      });
    }

    await index.upsert(vectors);
    res.status(200).json({ message: `Uppladdat ${vectors.length} chunk (Swish) framgångsrikt!` });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }
}
