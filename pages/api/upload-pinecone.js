import { Pinecone } from '@pinecone-database/pinecone';
import { v4 as uuid } from 'uuid';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const index = pinecone.Index('fortus-support');

    const chunks = [
      { keyword: 'Faktura, delbetalning', text: 'För att använda Faktura via Fortus som betalmetod för er webshop, POS eller affärssystem följ dessa steg: Kontakta Fortus. Ring in till oss på 010 – 222 15 20 eller skicka e-post via till support@fortuspay.com. Signera Fakturaköpsavtal. Avtal skickas till er via oneflow. När detta är signerat så sätts ni upp i systemet inom 24 timmar. Betalalternativ aktiveras i systemet. Fortus aktiverar fakturaknappen för er. När detta är klart kan ni börja skicka fakturor via Fortus till era kunder.' },
      { keyword: 'Swish', text: 'För att använda Swish som betalmetod för din webshop eller POS, följ dessa steg: Kontakta banken. Begär om att få Swish Handel kopplat till ditt företagskonto. Ange Fortus som teknisk leverantör när du pratar med banken. Fortus tekniska leverantörs ID är 9873196894. Uppge detta ID för att slutföra kopplingen. Automatisk aktivering. Efter att Swish är kopplat med Fortus som teknisk leverantör, kommer Fortus att automatiskt aktivera Swish i de valda försäljningskanalerna. När detta är klart kan du börja ta emot Swish-betalningar från dina kunder i de kanaler du har valt.' },
      { keyword: 'Dagsavslut', text: 'Steg för steg – Hur du gör ett dagsavslut i betalterminalen: Tryck på de tre prickarna längst upp till höger. Välj Dagsavslut. Tryck OK. Klart.' },
      { keyword: 'Retur', text: 'Steg för steg – Hur du gör en retur i betalterminalen: Tryck på de tre prickarna längst upp till höger. Välj Transaktionshistorik. Välj den rad du vill returnera, eller sök efter kvittonummer. Klicka på “Retur”. Markera den rad som skall returneras. Klicka på “Fortsätt”. Bekräfta retur genom att klicka på “Fortsätt”. Be kunden blippa eller sätta i kortet dit retur skall göras. Klart.' },
      { keyword: 'Hämta kopia på kvitto', text: 'Steg för steg – Hur du hämtar kopia på kvitto: Tryck på de tre prickarna längst upp till höger. Välj Transaktionshistorik. Leta upp eller sök efter kvitto. Tryck på kvittoikonen längst upp till höger. Klart.' },
      { keyword: 'Felsökning', text: 'Problem med betalterminal: Om ni har problem med betalterminalen och inte hittar lösningen på vår supportsida kan ni alltid kontakta oss via e-post eller telefon. E-post: support@fortuspay.com. Telefon: +46 10 222 15 20.' },
      { keyword: 'Fortus Web POS', text: 'Lägg till / Redigera kvittotexter och bild: Texter som redigeras visas i toppen och/eller foten på kvittot. Gå till: Butiker -> Hantera Butiker -> Välj Butik. Klicka på den butik du vill redigera. Scrolla ned till sektionen Inställningar kassa. Fyll i de texter du vill använda, t.ex. öppettider. Testa resultatet: Logga in på kassan. Sätt kassan i övningsläge. Genomför ett kontantköp för att se hur kvittolayouten ser ut. Tips för att formatera kvittotexten.' },
    ];

    const vectors = [];
    for (const chunk of chunks) {
      const embedResponse = await pinecone.inference.embed(
        'llama-text-embed-v2',
        [chunk.text],
        { input_type: 'passage' }  // Detta fixar felet för chunks!
      );
      const embedding = embedResponse.data[0].values;
      vectors.push({
        id: uuid(),
        values: embedding,
        metadata: { keyword: chunk.keyword, text: chunk.text }
      });
    }

    await index.upsert(vectors);
    res.status(200).json({ message: `Uppladdat ${vectors.length} chunks till Pinecone framgångsrikt!` });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message || 'Okänt fel vid uppladdning' });
  }
}
