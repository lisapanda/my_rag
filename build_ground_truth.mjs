import fs from "node:fs/promises";
import path from "node:path";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { Workbook } from "@oai/artifact-tool";

const sourceDir = String.raw`C:\Users\luyer\my_rag\data`;
const outputPath = path.join(sourceDir, "movie_wikipedia_ground_truth.csv");

const urls = {
  "2001_A_Space_Odyssey.pdf": "https://en.wikipedia.org/wiki/2001:_A_Space_Odyssey",
  "8½ - Wikipedia.pdf": "https://en.wikipedia.org/wiki/8%C2%BD",
  "Casablanca.pdf": "https://en.wikipedia.org/wiki/Casablanca_(film)",
  "Citizen_Kane.pdf": "https://en.wikipedia.org/wiki/Citizen_Kane",
  "Gone with the Wind (film) - Wikipedia.pdf": "https://en.wikipedia.org/wiki/Gone_with_the_Wind_(film)",
  "His Girl Friday - Wikipedia.pdf": "https://en.wikipedia.org/wiki/His_Girl_Friday",
  "Parasite_2019.pdf": "https://en.wikipedia.org/wiki/Parasite_(2019_film)",
  "Pulp_Fiction.pdf": "https://en.wikipedia.org/wiki/Pulp_Fiction",
  "Seven_Samurai.pdf": "https://en.wikipedia.org/wiki/Seven_Samurai",
  "Spirited_Away.pdf": "https://en.wikipedia.org/wiki/Spirited_Away",
  "The Narrow Margin - Wikipedia.pdf": "https://en.wikipedia.org/wiki/The_Narrow_Margin",
  "The_Dark_Knight.pdf": "https://en.wikipedia.org/wiki/The_Dark_Knight",
  "The_Godfather.pdf": "https://en.wikipedia.org/wiki/The_Godfather",
  "The_Matrix.pdf": "https://en.wikipedia.org/wiki/The_Matrix",
};

const headingWords = /^(plot|cast|casting|production|development|writing|screenplay|pre-production|filming|cinematography|editing|visual effects|special effects|music|soundtrack|themes|theme|analysis|interpretations|style|influences|release|theatrical release|home media|marketing|box office|reception|critical response|audience response|accolades|awards|legacy|influence|preservation|retrospective appraisal|controversies|historical accuracy|cultural impact|sequels|adaptations|remake|restoration|copyright|bibliography)$/i;
const rejectHeading = /^(references|external links|notes|further reading|see also|sources|footnotes)$/i;

function cleanText(text) {
  return text
    .replace(/\[[^\]]{1,12}\]/g, "")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .trim();
}

function titleFromFile(file) {
  return file.replace(/\.pdf$/i, "").replace(/ - Wikipedia$/i, "").replace(/_/g, " ");
}

async function extractPdf(filePath) {
  const bytes = new Uint8Array(await fs.readFile(filePath));
  const pdf = await getDocument({ data: bytes, disableWorker: true }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    let lastY = null;
    let line = "";
    const lines = [];
    for (const item of content.items) {
      const y = Math.round(item.transform?.[5] ?? 0);
      if (lastY !== null && Math.abs(y - lastY) > 2) {
        if (line.trim()) lines.push(line.trim());
        line = "";
      }
      line += (line ? " " : "") + item.str;
      lastY = y;
    }
    if (line.trim()) lines.push(line.trim());
    pages.push({ page: i, lines, text: cleanText(lines.join(" ")) });
  }
  return pages;
}

function detectSections(pages) {
  const sections = [{ name: "Overview", page: 1, text: "" }];
  let current = sections[0];
  for (const page of pages) {
    for (const raw of page.lines) {
      const line = cleanText(raw).replace(/^\d+(?:\.\d+)*\s+/, "");
      const headingLike = headingWords.test(line);
      if (headingLike && !rejectHeading.test(line) && !/^Wikipedia$/i.test(line)) {
        const existing = sections.find(s => s.name.toLowerCase() === line.toLowerCase());
        if (!existing) {
          current = { name: line, page: page.page, text: "" };
          sections.push(current);
          continue;
        }
      }
      current.text += " " + raw;
    }
  }
  return sections
    .map(s => ({ ...s, text: cleanText(s.text) }))
    .filter(s => s.text.length >= 140 && !rejectHeading.test(s.name));
}

function candidateSentences(text) {
  return cleanText(text)
    .split(/(?<=[.!?])\s+(?=[A-Z0-9"'])/)
    .map(cleanText)
    .filter(s => s.length >= 75 && s.length <= 360)
    .filter(s => /^[A-Z0-9"']/.test(s) && /[.!?]$/.test(s))
    .filter(s => !/https?:|Wikipedia|retrieved|ISBN|doi:/i.test(s))
    .filter(s => !/Theatrical release poster|Directed by .* Screenplay by|Production compan|Edited by|Running time|0:00|AFI's 100 Years/i.test(s))
    .filter(s => !/\.\.\.|— .*\d{4}/.test(s))
    .filter(s => (s.match(/[A-Za-z]/g) || []).length > 55);
}

function answerFor(title, section, text) {
  if (/^overview$/i.test(section)) {
    const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const lead = cleanText(text).match(new RegExp(`(${escaped}\\s+is (?:an?|the)\\b[^.]{55,320}\\.)`, "i"));
    if (lead) return lead[1];
  }
  const sentences = candidateSentences(text);
  if (!sentences.length) return null;
  const groups = [
    [/overview/i, [title.toLowerCase(), "film", "directed"]],
    [/plot/i, ["after", "when", "while", "finds", "discovers", "returns", "arrives"]],
    [/cast/i, ["cast", "role", "actor", "actress", "portray", "starring"]],
    [/music|soundtrack/i, ["music", "score", "composer", "soundtrack", "song"]],
    [/box office/i, ["gross", "box office", "million", "revenue", "ticket"]],
    [/reception|response|appraisal/i, ["critic", "review", "praised", "acclaim", "rating"]],
    [/award|accolade/i, ["award", "won", "nomination", "nominated", "prize"]],
    [/legacy|influence|impact|preservation/i, ["influence", "legacy", "registry", "preservation", "greatest", "inspired"]],
    [/release|media|marketing/i, ["release", "premiere", "theater", "campaign", "dvd", "blu-ray"]],
    [/production|development|writing|screenplay|filming|cinematography|editing|effects/i, ["production", "filming", "shot", "camera", "script", "screenplay", "studio", "effect"]],
    [/theme|analysis|interpretation|style/i, ["theme", "interpret", "symbol", "style", "meaning", "analysis"]],
  ];
  const match = groups.find(([re]) => re.test(section));
  const keys = match ? match[1] : [];
  let pool = sentences;
  const lowerSection = section.toLowerCase();
  const strict = {
    plot: /\b(after|when|while|finds|discovers|kills|returns|arrives|tells|takes|reveals|escapes|refuses|agrees|leaves|meets|shoots|fights|warns)\b/i,
    cast: /\b(cast|role|actor|actress|portray|offered|audition|played|starring)\b/i,
    music: /\b(music|score|composer|soundtrack|song|theme|album)\b/i,
    release: /\b(release|released|premiere|theater|theatre|dvd|blu-ray|campaign|screen)\b/i,
    reception: /\b(critic|review|rating|acclaim|approval|metacritic|rotten tomatoes)\b/i,
    award: /\b(award|won|nominated|nomination|prize|palme d'or)\b/i,
    legacy: /\b(legacy|influence|inspired|registry|preservation|selected|ranked|greatest)\b/i,
    theme: /\b(theme|symbol|represent|class|identity|meaning|interpret)\b/i,
    box: /\b(gross|box office|million|billion|revenue|ticket)\b/i,
    production: /\b(production|filming|shot|studio|script|screenplay|camera|budget|set|constructed|director|effect)\b/i,
  };
  let required = null;
  if (lowerSection.includes("plot")) required = strict.plot;
  else if (lowerSection.includes("cast")) required = strict.cast;
  else if (/music|soundtrack/.test(lowerSection)) required = strict.music;
  else if (/box office/.test(lowerSection)) required = strict.box;
  else if (/reception|response|appraisal/.test(lowerSection)) required = strict.reception;
  else if (/award|accolade/.test(lowerSection)) required = strict.award;
  else if (/legacy|influence|impact|preservation/.test(lowerSection)) required = strict.legacy;
  else if (/release|media|marketing/.test(lowerSection)) required = strict.release;
  else if (/theme|analysis|interpretation|style/.test(lowerSection)) required = strict.theme;
  else if (/production|development|writing|screenplay|filming|cinematography|editing|effects/.test(lowerSection)) required = strict.production;
  if (required) pool = sentences.filter(s => required.test(s));
  if (lowerSection.includes("plot")) pool = pool.filter(s => !/\b(actor|actress|cast|role|filming|production|screenplay|director)\b/i.test(s));
  if (!pool.length) return null;
  const selected = [...pool].sort((a, b) => {
    const score = s => keys.reduce((n, k) => n + (s.toLowerCase().includes(k) ? 3 : 0), 0)
      + (s.length >= 100 && s.length <= 260 ? 2 : 0)
      - (/^[A-Z][a-z]+\s+(with|by|and)\b/.test(s) ? 2 : 0);
    return score(b) - score(a);
  })[0];
  return selected.replace(/^(?:(?:Plot|Cast|Casting|Production|Development|Writing|Screenplay|Filming|Cinematography|Editing|Visual effects|Special effects|Music|Soundtrack|Themes|Analysis|Style|Release|Marketing|Box office|Reception|Critical response|Accolades|Awards|Legacy|Influence)\s+){2,}/i, "");
}

function factQA(title, section, rawAnswer, index) {
  let answer = cleanText(rawAnswer);
  const conditional = answer.match(/^(When|After|While) (.{15,180}?), ([A-Z][\s\S]{20,220})\.$/);
  if (conditional) {
    return { question: `What happens ${conditional[1].toLowerCase()} ${conditional[2]}?`, answer: conditional[3] + "." };
  }
  const s = section.toLowerCase();
  if (s === "overview") {
    const intro = answer.match(/\bis a ([^.]*?\bfilm)\b/i);
    if (intro) return { question: `What year and genre describe ${title}?`, answer: `It is a ${intro[1]}.` };
    return { question: `What is ${title}?`, answer };
  }
  const rt = answer.match(/Rotten Tomatoes[^.]*?(\d+%)[^.]*?(?:based on ([\d,]+) reviews)?/i);
  if (rt) return { question: `What Rotten Tomatoes rating is reported for ${title}?`, answer: rt[0].replace(/^[^,]*,?\s*/, "").trim() + "." };
  const mc = answer.match(/Metacritic[^.]*?(\d+ out of 100)[^.]*?(?:based on reviews from ([\d,]+) critics)?/i);
  if (mc) return { question: `What Metacritic score is reported for ${title}?`, answer: mc[0].trim() + "." };
  const gross = answer.match(/\b(?:grossed|box office (?:to|of))\s+([^.;]{3,90})/i);
  if (gross) return { question: `What box-office result is reported for ${title}?`, answer: gross[1].replace(/\s+/g," ").trim() + "." };
  const directed = answer.match(/\bdirected by ([A-Z][A-Za-z .'-]+?)(?:,| and|\.)/);
  if (directed) return { question: `Who directed ${title}?`, answer: directed[1].trim() + "." };
  const composer = answer.match(/(?:composer|composed by|scored by|hired .*? composer)\s+([A-Z][A-Za-z .'-]+?)(?:,| to| who|\.)/i);
  if (composer) return { question: `Who composed or scored the music for ${title}?`, answer: composer[1].trim() + "." };
  if (/\bpremiered\b/i.test(answer)) return { question: `When and where did ${title} premiere?`, answer };
  if (/\breleased\b/i.test(answer) && /\b(?:19|20)\d{2}\b/.test(answer)) return { question: `When was the cited release of ${title}?`, answer };
  if (/\bshot\b/i.test(answer) && /\b(?:in|at|on)\b/i.test(answer)) return { question: `Where or under what conditions was ${title} shot?`, answer };
  if (/\b(?:won|nominated|award)\b/i.test(answer)) return { question: `Which award or nomination is stated for ${title}?`, answer };
  if (/\b(?:selected|preservation|registry)\b/i.test(answer)) return { question: `What preservation or registry recognition did ${title} receive?`, answer };
  if (s.includes("plot")) return { question: `What occurs in this plot event from ${title}?`, answer };
  if (s.includes("cast")) return { question: `Which specific casting fact is reported for ${title}?`, answer };
  if (/music|soundtrack/.test(s)) return { question: `Which specific music or soundtrack fact is reported for ${title}?`, answer };
  if (/theme|analysis|interpretation|style/.test(s)) return { question: `Which theme, symbol, or stylistic interpretation is identified in ${title}?`, answer };
  if (/production|development|writing|screenplay|filming|cinematography|editing|effects/.test(s)) return { question: `Which specific production fact is reported for ${title} in the ${section} section?`, answer };
  if (/release|media|marketing/.test(s)) return { question: `Which specific release or marketing fact is reported for ${title}?`, answer };
  if (/reception|response|appraisal/.test(s)) return { question: `Which critic or review score is cited for ${title}?`, answer };
  if (/legacy|influence|impact|preservation/.test(s)) return { question: `Which specific legacy or influence is identified for ${title}?`, answer };
  return { question: `Which factual statement is given for ${title} on page ${section.match(/\d+/)?.[0] || "cited"}?`, answer };
}

function csvEscape(value) {
  const s = String(value ?? "");
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

const files = (await fs.readdir(sourceDir)).filter(f => f.toLowerCase().endsWith(".pdf")).sort();
if (files.length === 0) throw new Error("No PDFs found");

const docs = [];
for (const file of files) {
  const pages = await extractPdf(path.join(sourceDir, file));
  const sections = detectSections(pages);
  const articleTitle = cleanText(pages[0]?.lines?.[0] || titleFromFile(file));
  docs.push({ file, title: articleTitle, pages, sections });
}

const target = 100;
const quotas = new Map(docs.map((d, i) => [d.file, 7 + (i < 2 ? 1 : 0)]));
const rows = [];
for (const doc of docs) {
  const quota = quotas.get(doc.file);
  const ranked = [...doc.sections].sort((a, b) => {
    const priority = x => /^(overview|plot|cast|production|development|filming|music|release|box office|reception|critical response|accolades|legacy|themes|influence)$/i.test(x.name) ? 0 : 1;
    return priority(a) - priority(b) || a.page - b.page;
  });
  const usedQuestions = new Set();
  for (const section of ranked) {
    if (rows.filter(r => r.source_file === doc.file).length >= quota) break;
    const pageWindow = doc.pages
      .slice(Math.max(0, section.page - 1), Math.min(doc.pages.length, section.page + 2))
      .map(p => p.text).join(" ");
    const answer = answerFor(doc.title, section.name, section.text + " " + pageWindow);
    if (!answer) continue;
    const qa = factQA(doc.title, section.name, answer, rows.length);
    let question = qa.question;
    if (usedQuestions.has(question)) question = question.replace("?", ` (detail ${usedQuestions.size + 1})?`);
    usedQuestions.add(question);
    rows.push({
      id: rows.length + 1,
      question,
      answer: qa.answer,
      source_file: doc.file,
      article_title: doc.title,
      section: section.name,
      page: section.page,
      source_url: urls[doc.file] || "",
      answer_type: "extractive",
    });
  }
  let pageCursor = 1;
  while (rows.filter(r => r.source_file === doc.file).length < quota && pageCursor <= doc.pages.length) {
    const page = doc.pages[pageCursor - 1];
    const answer = candidateSentences(page.text)[0];
    if (answer) {
      const section = `Page ${page.page} supplemental coverage`;
      const qa = factQA(doc.title, section, answer, rows.length);
      rows.push({ id: rows.length + 1, question: qa.question, answer: qa.answer,
        source_file: doc.file, article_title: doc.title, section, page: page.page,
        source_url: urls[doc.file] || "", answer_type: "extractive" });
    }
    pageCursor++;
  }
}

if (rows.length !== target) throw new Error(`Expected ${target} rows, generated ${rows.length}`);
const curated = {
  1:["What year was 2001: A Space Odyssey released, and what genre is it?","It is a 1968 epic science fiction film."],
  2:["Which Academy Award did Stanley Kubrick win for 2001: A Space Odyssey?","He won for directing the film's visual effects."],
  3:["What does Bowman do after HAL reports that the antenna control device will fail?","He retrieves the device in an EVA pod but finds nothing wrong."],
  4:["Which studio financed and distributed 2001: A Space Odyssey?","Metro-Goldwyn-Mayer (MGM)."],
  5:["When was filming of the actors for 2001: A Space Odyssey completed?","September 1967."],
  6:["Who composed the original score that Kubrick ultimately did not use for 2001: A Space Odyssey?","Alex North."],
  7:["What Rotten Tomatoes rating is reported for 2001: A Space Odyssey?","90% based on 166 reviews, with an average rating of 9.6/10."],
  8:["Which filmmakers discuss the influence of 2001: A Space Odyssey in the featurette Standing on the Shoulders of Kubrick?","Steven Spielberg, George Lucas, and others."],
  9:["What year and genre describe 8½, and who directed it?","It is a 1963 Italian avant-garde comedy-drama directed by Federico Fellini."],
  10:["Whom does Guido hire at the spa to review his film ideas in 8½?","A well-known critic, who then harshly criticizes the ideas."],
  11:["Which lead actors had Fellini cast when he began production of 8½ in spring 1962?","Marcello Mastroianni, Anouk Aimée, and Sandra Milo."],
  12:["What theme did Tullio Pinelli associate with the abandoned train sequence in 8½?","Suicide."],
  13:["What Metacritic score is reported for 8½?","93 out of 100, indicating universal acclaim."],
  14:["Where did the Writers Guild of America West rank the screenplay of 8½ in 2006?","87th on its list of 101 Greatest Screenplays."],
  15:["What creative problem does Guido Anselmi face in 8½?","He suffers from writer's block while trying to direct an epic science fiction film."],
  16:["Why does Claudia consider the protagonist of Guido's proposed film unsympathetic?","She says he is incapable of love."],
  17:["What year and genre describe Casablanca?","It is a 1942 American romantic drama film."],
  18:["What is the difference in speaking parts between the Casablanca film script and the original play?","The film script had 22 speaking parts, compared with 16 in the play."],
  19:["Why had Ilsa believed her husband was dead when she fell in love with Rick in Paris?","She believed he had been killed while escaping from a concentration camp."],
  20:["Which actresses were considered for the role of Ilsa in Casablanca?","Ann Sheridan, Hedy Lamarr, Luise Rainer, and Michèle Morgan."],
  21:["In what order was Casablanca filmed, and why?","It was shot in sequence because only the first half of the script was ready when filming began."],
  22:["When was Casablanca's first original soundtrack album released on compact disc?","In 1997, almost 55 years after the film's premiere."],
  23:["When and where did Casablanca premiere?","At the Hollywood Theater in New York City on November 26, 1942."],
  24:["What year and genre describe Citizen Kane?","It is a 1941 American drama film."],
  25:["Which Academy Award did Citizen Kane win?","Best Writing (Original Screenplay), awarded to Herman J. Mankiewicz and Orson Welles."],
  26:["Why does Kane fire Leland from the Inquirer?","Leland wrote a negative review of Susan's disastrous opera debut."],
  27:["What small role does Alan Ladd play in Citizen Kane?","A pipe-smoking reporter near the end of the film."],
  28:["What unusual contractual privilege did Orson Welles receive for Citizen Kane?","Final cut privilege."],
  29:["Which post-production technique was Citizen Kane deliberately filmed to support?","Slow dissolves."],
  30:["Who composed the score for Citizen Kane?","Bernard Herrmann."],
  31:["What year and genre describe Gone with the Wind?","It is a 1939 American epic historical romance film."],
  32:["Why was the start of filming Gone with the Wind delayed for two years?","David O. Selznick was determined to secure Clark Gable for the role of Rhett Butler."],
  33:["What happens after Rhett returns from an extended trip to London?","Scarlett tells him she is pregnant; during the ensuing argument she falls down stairs and loses the baby."],
  34:["Which role was promoted through a major casting search for Gone with the Wind?","Scarlett O'Hara."],
  35:["Where were most location scenes for Gone with the Wind photographed?","In California, mainly Los Angeles County and neighboring Ventura County."],
  36:["Which plantation is associated with the best-known musical theme from Gone with the Wind?","Tara, the O'Hara plantation."],
  37:["What historical anniversary did the 1961 rerelease of Gone with the Wind commemorate?","The centennial of the start of the American Civil War."],
  38:["What year and genre describe His Girl Friday?","It is a 1940 American screwball comedy film."],
  39:["Who wrote the screenplay for His Girl Friday, and what major change was made to the source material?","Charles Lederer wrote it, and Hildy Johnson was changed from a man to a woman."],
  40:["Which roles did Harry Cohn initially intend Cary Grant and Walter Winchell to play in His Girl Friday?","Grant was to play the reporter and Winchell the editor."],
  41:["What amount of time did the one-camera scene in His Girl Friday take to film?","Four days instead of the intended two."],
  42:["When and where did His Girl Friday premiere?","At Radio City Music Hall in New York City on January 11, 1940."],
  43:["What Rotten Tomatoes approval rating is reported for His Girl Friday?","99% based on 106 reviews."],
  44:["What preservation recognition did His Girl Friday receive in 1993?","The Library of Congress selected it for the United States National Film Registry."],
  45:["Who directed Parasite, and in what year was it released?","Bong Joon-ho directed the 2019 film."],
  46:["What did Han Jin-won produce from Bong Joon-ho's 15-page treatment for Parasite?","Three different drafts of the screenplay."],
  47:["Who unexpectedly arrives while the Kim family is enjoying the Park house in Parasite?","Moon-gwang, the former housekeeper."],
  48:["Who plays Kim Ki-taek in Parasite?","Song Kang-ho."],
  49:["Which parts of the Park house in Parasite were created in post-production?","Everything above the first floor; the physical house itself was constructed as a set."],
  50:["What English title was used for the Parasite soundtrack song later submitted for Oscar consideration?","Soju One Glass."],
  51:["When was Parasite released on Blu-ray and DVD by Neon?","January 28, 2020."],
  52:["What year and genre describe Pulp Fiction?","It is a 1994 American crime film."],
  53:["For which actor did Quentin Tarantino write the role of Winston Wolfe in Pulp Fiction?","Harvey Keitel."],
  54:["What does Mia do with Vincent's heroin while he is in the bathroom in Pulp Fiction?","She mistakes it for cocaine and snorts it."],
  55:["Which actors were offered the role of Marsellus Wallace before Ving Rhames was cast?","Max Julien and Sid Haig."],
  56:["What kind of music did Quentin Tarantino use instead of an original score for Pulp Fiction?","An eclectic mix of surf music, rock and roll, soul, and pop songs."],
  57:["What Rotten Tomatoes approval rating is reported for Pulp Fiction?","92% based on 185 critics' reviews."],
  58:["Which two elements from Roger Avary's True Romance screenplay were reused in Pulp Fiction?","The miraculous missed shots by a hidden gunman and the rear-seat automobile killing."],
  59:["What year and genre describe Seven Samurai, and who directed it?","It is a 1954 Japanese epic samurai action drama directed by Akira Kurosawa."],
  60:["Who directed Seven Samurai?","Akira Kurosawa.","Overview",1],
  61:["Which surviving samurai stand before their comrades' burial mounds at the end of Seven Samurai?","Kambei, Katsushirō, and Shichirōji."],
  62:["What prior sword experience did Seiji Miyaguchi have when cast as the expert swordsman Kyūzō?","He had never used or carried a sword."],
  63:["Why did Kurosawa avoid filming the final battle early in the production of Seven Samurai?","He believed the studio would have forced him to stop production afterward."],
  64:["Who composed the Samurai Theme for Seven Samurai?","Fumio Hayasaka."],
  65:["Which musical technique does Yoshio Kobayashi identify as a major theme of Seven Samurai's score?","The leitmotif."],
  66:["What year and genre describe Spirited Away?","It is a 2001 Japanese animated fantasy film."],
  67:["Whom does Chihiro meet near the bathhouse in Spirited Away?","Haku, who warns her to cross the riverbed before sunset."],
  68:["Which Japan Gold Disk Award did the Spirited Away soundtrack receive?","Animation Album of the Year at the 17th Japan Gold Disk Awards."],
  69:["What does the bathhouse symbolize in Spirited Away?","Japan's distorted cultural identity."],
  70:["When did Walt Disney Studios Japan release Spirited Away on Blu-ray?","July 14, 2014."],
  71:["What Metacritic score is reported for Spirited Away?","96 out of 100 based on 41 critics, indicating universal acclaim."],
  72:["Which Academy Award did Spirited Away win after Disney campaigned for the film?","Best Animated Feature."],
  73:["What year and genre describe The Narrow Margin?","It is a 1952 American film noir thriller."],
  74:["What technique created the train-window backgrounds in The Narrow Margin?","The train scenes were shot on an RKO soundstage using rear projection."],
  75:["Whom does Brown turn Kemp over to after their fight in The Narrow Margin?","Railroad agent Sam Jennings."],
  76:["Who directed the 1990 remake of The Narrow Margin?","Peter Hyams.","Remake",4],
  77:["What phrase did New York Times critic Howard Thompson use to describe The Narrow Margin?","A trim, sizzling little humdinger.","Critical response",3],
  78:["What assignment drives the plot of The Narrow Margin?","A police detective must protect an important witness on a cross-country train."],
  79:["What budget is listed for The Narrow Margin?","$230,000.","Overview",1],
  80:["What year and genre describe The Dark Knight?","It is a 2008 superhero film."],
  81:["What duration did Christopher and Jonathan Nolan spend collaborating on the final script of The Dark Knight?","Six months during pre-production."],
  82:["Where does Batman capture Lau in The Dark Knight?","Hong Kong."],
  83:["Why did Christopher Nolan want a Joker actor who could withstand intense scrutiny?","Because the performance would be compared with Jack Nicholson's popular portrayal."],
  84:["Who composed the score for The Dark Knight?","Hans Zimmer and James Newton Howard."],
  85:["When was The Dark Knight widely released in the United States and Canada?","July 18, 2008."],
  86:["What domestic total did The Dark Knight reach before leaving theaters in March 2009?","$533.3 million."],
  87:["What year and genre describe The Godfather?","It is a 1972 American epic gangster film."],
  88:["What price did Paramount Pictures pay for the film rights to The Godfather novel?","$80,000.","Overview",1],
  89:["What causes Captain McCluskey to back down outside the hospital in The Godfather?","Tom Hagen arrives with additional bodyguards and asserts his legal authority."],
  90:["Which two roles caused major casting disagreements during production of The Godfather?","Vito Corleone and Michael Corleone."],
  91:["Where did Paramount executives initially want The Godfather to be set and filmed?","In contemporary Kansas City, using the studio backlot."],
  92:["What became the focus of post-production after filming The Godfather ended on August 7?","Trimming the film to a manageable length."],
  93:["Who composed the principal score for The Godfather?","Nino Rota."],
  94:["What year and genre describe The Matrix?","It is a 1999 science fiction action film."],
  95:["What visual effect did The Matrix popularize?","Bullet time, which shows slowed action while the camera appears to move at normal speed."],
  96:["Who betrays the crew to the Agents in The Matrix?","Cypher."],
  97:["Which actor turned down the role of Neo and later said he was not mature enough for it?","Will Smith."],
  98:["What financial risk did the studio take when making The Matrix?","It invested $60 million in a script by relatively inexperienced directors using unproven special effects."],
  99:["What action effect did the Wachowskis describe in the screenplay for The Matrix?","An action sequence in which time slows while the camera rapidly pivots around the subjects."],
  100:["What opening gross is reported for The Matrix in Taiwan?","$1.8 million, the third-highest opening there at the time."]
};
for (const row of rows) {
  const item = curated[row.id];
  if (!item) throw new Error(`Missing curated QA for row ${row.id}`);
  row.question = item[0];
  row.answer = item[1];
  if (item[2]) row.section = item[2];
  if (item[3]) row.page = item[3];
}
rows.forEach((row, i) => { row.id = i + 1; });
const forbiddenQuestion = /^(how\b|according to\b)|\b(detail|key fact|factual statement|what occurs in this plot event)\b/i;
const badQuestions = rows.filter(r => forbiddenQuestion.test(r.question));
if (badQuestions.length) throw new Error(`Vague questions remain: ${badQuestions.map(r => r.id).join(",")}`);
if (new Set(rows.map(r => r.question.toLowerCase())).size !== rows.length) throw new Error("Duplicate questions remain");
if (rows.some(r => !r.answer.trim() || !r.question.trim())) throw new Error("Blank question or answer remains");

const stop = new Set("a an and are as at be because by did do does for from had has have he her him his i in into is it its of on or she that the their them they this to was were what when where which who why with would".split(" "));
const tokens = s => cleanText(s).toLowerCase().match(/[a-z0-9]+/g)?.filter(x => x.length > 2 && !stop.has(x)) || [];
for (const row of rows) {
  const doc = docs.find(d => d.file === row.source_file);
  const ts = [...new Set(tokens(row.answer))];
  let best = { page: Number(row.page), score: -1 };
  for (const page of doc.pages) {
    const text = page.text.toLowerCase();
    const score = ts.length ? ts.filter(t => text.includes(t)).length / ts.length : 0;
    if (score > best.score) best = { page: page.page, score };
  }
  if (best.score >= 0.5) row.page = best.page;
}
const grounding = rows.map(row => {
  const doc = docs.find(d => d.file === row.source_file);
  const pageText = doc.pages.slice(Math.max(0, Number(row.page) - 2), Math.min(doc.pages.length, Number(row.page) + 1)).map(p => p.text.toLowerCase()).join(" ");
  const ts = [...new Set(tokens(row.answer))];
  const matched = ts.filter(t => pageText.includes(t));
  return { id: row.id, score: ts.length ? matched.length / ts.length : 0, missing: ts.filter(t => !pageText.includes(t)).slice(0, 8) };
});
const lowGrounding = grounding.filter(x => x.score < 0.55).sort((a,b) => a.score-b.score);
if (lowGrounding.length > 10) throw new Error(`Too many weakly grounded rows: ${JSON.stringify(lowGrounding)}`);
const headers = ["id","question","answer","source_file","article_title","section","page","source_url","answer_type"];
const csv = [headers.join(","), ...rows.map(r => headers.map(h => csvEscape(r[h])).join(","))].join("\r\n") + "\r\n";

const workbook = await Workbook.fromCSV(csv, { sheetName: "Ground Truth" });
const inspected = await workbook.inspect({ kind: "table", range: "Ground Truth!A1:I8", include: "values", tableMaxRows: 8, tableMaxCols: 9, maxChars: 6000 });
await fs.writeFile(outputPath, "\uFEFF" + csv, "utf8");

const coverage = Object.fromEntries(files.map(f => [f, rows.filter(r => r.source_file === f).length]));
console.log(inspected.ndjson);
console.log(JSON.stringify({ outputPath, pdfCount: files.length, rowCount: rows.length, coverage, lowGrounding }, null, 2));
