# Chat Robot

## Opis Problema

Glavna ideja ovog projekta je kreiranje chat robota pomoću veštačke inteligencije. Cilj je obučiti robota da automatski odgovara na jednostavnija pitanja u formi Q&amp;A. Jedna od poželjnih osobina je oponašanje odgovora neke specifične ličnosti. U tu svrhu, chat robot se može iskoristiti kao chat agent za razne stvari (sales support, IT support, Q&amp;A websites) a da korisnik i ne primeti da razgovara sa robotom.

## Opis Podataka

Dataset na kom je obučavan chat robot je Facebook chat history sa privatnog profila. Inicijalna ideja je bila da robot oponaša jednog facebook korisnika. Može se koristiti bilo koji drugi tekstualni dataset koji sledi sledeći format:

- --Svaka neparna linija je input (Question)
- --Svaka parna linija je output (Answer)

## Prikupljanje i priprema podataka

Kopija Facebook podataka može da se preuzme sa njihovog sajta u zip formatu iz Podešavanja na facebook nalogu. Ceo chat history se nalazi u jednom .htm fajlu koji je netrivijalan za parsiranje. Nakon par sati neuspešnih pokušaja da se isparsiraju poruke i sačuvaju u obični .txt fajl, jednostavna google pretraga je pomogla u traženju rešenja.

[https://github.com/ownaginatious/fbchat-archive-parser](https://github.com/ownaginatious/fbchat-archive-parser)

Komanda:

_&gt;&gt;&gt; fbcap messages.htm -t &quot;Ime Korisnika&quot; &gt; konverzacija.txt_

Je odradila mukotrpan posao u par sekundi i generisala tekst fajl koji je jako lepo struktuiran. Za potrebu ovog projekta generisana su dva dataseta iz dve različite konverzacije koje su se odvojeno obučavale. Naravno, bilo je potrebno odraditi dodatno formatiranje (uklanjanje nepotrebnih razmaka, praznih linija, timestamp poruka itd). Finalni format poruka izgleda ovako:

_…_

_Ime\_Korisnika1: Dodji na pivo_

_Ime\_Korisnika2: Eto me_

_…_

Druga faza parsiranja je odrađena u Pythonu. Za detaljnije objašnjenje pogledati parse.py u git repozitorijumu. Ova faza obuhvata:

1. Konvertovanje svih slova u lower case.
2. Uklanjanje akcentovanih slova &quot;čćžšđ&quot;.
3. Izbacivanje nekih specijalnih karaktera.
4. Odvajanje interpunkcijskih znakova sa razmakom da bi se stvorili odvojeni tokeni.
5. Spajanje više uzastopnih poruka jednog korisnika u jedno pitanje, odnosno odgovor.
6. Svaka kombinacija &quot;hahaha&quot; je pretvorena u &quot;haha&quot;.

Treća faza parsiranja je stvaranje tokena za svaku reč koja se učestalo pojavljuje. Postoji nekoliko predefinisanih tokena:

- \_UNK - za reči koje se ne pojavljuju često.
- \_EOS - End of string za označavanje kraja rečenice.
- \_GO - Indikator za početak generisanja odgovora.
- \_PAD - Prazno mesto zbog fiksne veličine ulaza.

Primer tokenizovane rečenice gde je broj ulaza fiksiran na 8:

&gt;&gt;&gt; Dodji na pivo.

[&#39;dodji&#39;, &#39;na&#39;, &#39;pivo&#39;, &#39;.&#39;, &#39;\_EOS&#39;, &#39;\_PAD&#39;, &#39;\_PAD&#39;, &#39;\_PAD&#39;]

Odgovor:

&gt;&gt;&gt; Eto me

[&#39;\_GO&#39;, &#39;eto&#39;, &#39;me&#39;, &#39;.&#39;, &#39;\_EOS&#39;, &#39;\_PAD&#39;, &#39;\_PAD&#39;, &#39;\_PAD&#39;]

Veličina rečnika je postavljena na 11500 reči kako bi postotak nepoznatih reči bio ispod 2%. Svaka od 11500 reči je dobila svoj jedinstveni indeks koji je korišten za obučavanje neuronske mreže. Neuronska mreža koristi samo indekse reči za obučavanje i na osnovu toga uspeva da nauči vezu između različitih indeksa (reči).

Dataset nakon parsiranja je zauzimao 1142KB.

## Opis Neuronske Mreže

Rekurentna neuronska mreža (RNN) je empirijski pokazana kao najbolje rešenje za modelovanje prirodnih jezika. Pošto je ovo problem koji zahteva ulaz promenjive veličine, odnosno niz ulaza i niz izlaza, RNN mreža mora da podrži više vektora i na ulazi i izlazu. Mreža na kojoj se obučavao chat robot se sastoji iz dve rekurentne mreže: Encoder koji obrađuje ulaz i Decoder koji generiše izlaz na osnovu izlaza iz Encodera. Ovaj model je poznatiji pod nazivom &quot;Sequence-to-Sequence&quot; (više detalja u referencama).

## Obučavanje mreže

Modelovanje i obučavanje je vršeno pomoću TensorFlow biblioteke. Inicijalna arhitektura mreže se sastojala od 1024 LSTM RNN ćelije po sloju u 3 sloja. LSTM (Long Short-Term Memory) je arhitektura koja omogućava internu memoriju neurona koja može da zapamti nešto što je bilo na inputu i pre dosta koraka. Takozvani &quot;long-term dependencies&quot;.

Jedna iteracija obučavanja je trajala poprilično dugo pa je finalna arhitektura pojednostavljena na 364 LSTM RNN ćelije po sloju i 2 sloja.

Obučavanje je testirano i na GRU (Gated Recurrent Unit) ćelijama koje su računski jednostavnije i brže konvergiraju ka rešenju. Međutim, &quot;odgovori&quot; nisu bili baš nešto preterano smisleni tako da je finalna mreža obučavana sa LSTM ćelijama.

## Rezultati

Nakon 8000 iteracija na datasetu od 1142KB rezultati su bolji nego očekivani pri korišćenju LSTM ćelija.

q: [sta radis]; a: [nistaa smor brate]

q: [cao]; a: [e cao :d]

q: [kravo jedna]; a: [volim i ja tebe :p ajmo na pivo :d]

q: [kad ces u ns ?]; a: [dok ne ustanem ne znam mozda pila 12]

q: [sta delas]; a: [ma smaram se skoro sam zaspala ali se ne odjavljujem :d sta ti?]

q: [&lt;3]; a: [e bosno moja  &lt;3]

q: [dobio sam 10]; a: [znala sam :d sta ti ti ? :d]

q: [hocu jagodu]; a: [ok . :d]

(Odabrani zanimljivi odgovori)

## GPU vs CPU Computing

TensorFlow podržava obučavanje na NVIDIA grafičkim karticama novije generacije koje poseduju određenu verziju CUDA Compatibility (&gt;=3.5 trenutno). Brzina obučavanja na GPU u odnosu na procesor je 12.5x na testiranom hardveru.

**Procesor:** i7-3537u, 2 cores - 4 threads, 2.0GHz

**GPU:** nVidia GTX 960m, 4GB DDR5, 640 CUDA cores, 1000MHz clock speed

Trajanje jedne iteracije na procesoru je oko ~3000ms, dok je na GPU  ~240ms pri čemu je GPU Load oko 70%. Obučavanje od 8000 iteracija je trajalo 32min na GPU a na procesoru bi trajalo oko 7h.

## Zaključak

Kao i u svakom chatu, postojalo je mnogo grešaka i mnogo istih reči koji su različito napisani. S obzirom na relativno kratko obučavanje relativno malog (i nepravilnog) dataseta, rezultati su bolji nego očekivani. Mreža je uspela da nauči korelaciju između reči i da odgovara u sličnom ili istom kontekstu u kom je i pitanje. Fascinantno je napomenuti da mreža inicijalno nema pojam o tome šta je reč i rečenica. Obučavanje se vrši na indeksima iz rečnika koji je generisan pri trećoj fazi parsiranja i sve što mreža na ulaz prima su Integeri koji predstavljaju indeks iz rečnika [0, 11500].

Trenutno, chat robot ne vodi evidenciju o prethodnim porukama i ne može da vodi konverzaciju. Nastavak ovog projekta bi bio unapređenje chat robota da vodi konverzaciju kroz više poruka.

Reference

[http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97\_lstm.pdf](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

[https://www.tensorflow.org/tutorials/seq2seq](https://www.tensorflow.org/tutorials/seq2seq)

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[https://github.com/ownaginatious/fbchat-archive-parser](https://github.com/ownaginatious/fbchat-archive-parser)

[https://github.com/suriyadeepan/practical\_seq2seq](https://github.com/suriyadeepan/practical_seq2seq)

[http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[https://arxiv.org/abs/1409.3215](https://arxiv.org/abs/1409.3215)