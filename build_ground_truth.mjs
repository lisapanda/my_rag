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
  const selected = [...sentences].sort((a, b) => {
    const score = s => keys.reduce((n, k) => n + (s.toLowerCase().includes(k) ? 3 : 0), 0)
      + (s.length >= 100 && s.length <= 260 ? 2 : 0)
      - (/^[A-Z][a-z]+\s+(with|by|and)\b/.test(s) ? 2 : 0);
    return score(b) - score(a);
  })[0];
  return selected.replace(/^(?:(?:Plot|Cast|Casting|Production|Development|Writing|Screenplay|Filming|Cinematography|Editing|Visual effects|Special effects|Music|Soundtrack|Themes|Analysis|Style|Release|Marketing|Box office|Reception|Critical response|Accolades|Awards|Legacy|Influence)\s+){2,}/i, "");
}

function questionFor(title, section, answer, index) {
  const s = section.toLowerCase();
  if (s.includes("plot")) return `What important plot event involving the film's characters is described in ${title}?`;
  if (s === "overview") return `How does the article introduce ${title}?`;
  if (s.includes("cast")) return `What casting detail is given for ${title}?`;
  if (/music|soundtrack/.test(s)) return `What does the article say about the music or soundtrack of ${title}?`;
  if (/box office/.test(s)) return `What box-office detail is reported for ${title}?`;
  if (/reception|response|appraisal/.test(s)) return `How does the article characterize the reception of ${title}?`;
  if (/award|accolade/.test(s)) return `What award or accolade detail is reported for ${title}?`;
  if (/legacy|influence|impact|preservation/.test(s)) return `What legacy or influence does the article attribute to ${title}?`;
  if (/release|media|marketing/.test(s)) return `What release or marketing detail is given for ${title}?`;
  if (/production|development|writing|screenplay|filming|cinematography|editing|effects/.test(s)) return `What production detail does the ${section} section give about ${title}?`;
  if (/theme|analysis|interpretation|style/.test(s)) return `What interpretive or stylistic point does the ${section} section make about ${title}?`;
  const lead = answer.split(" ").slice(0, 7).join(" ").replace(/[,:;]$/, "");
  return index % 2 === 0
    ? `According to the ${section} section, what does the article explain about ${lead}?`
    : `What key fact is stated in the ${section} section of the article on ${title}?`;
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
    let question = questionFor(doc.title, section.name, answer, rows.length);
    if (usedQuestions.has(question)) question = question.replace("?", ` (detail ${usedQuestions.size + 1})?`);
    usedQuestions.add(question);
    rows.push({
      id: rows.length + 1,
      question,
      answer,
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
      rows.push({ id: rows.length + 1, question: questionFor(doc.title, section, answer, rows.length), answer,
        source_file: doc.file, article_title: doc.title, section, page: page.page,
        source_url: urls[doc.file] || "", answer_type: "extractive" });
    }
    pageCursor++;
  }
}

if (rows.length !== target) throw new Error(`Expected ${target} rows, generated ${rows.length}`);
rows.forEach((row, i) => { row.id = i + 1; });
const headers = ["id","question","answer","source_file","article_title","section","page","source_url","answer_type"];
const csv = [headers.join(","), ...rows.map(r => headers.map(h => csvEscape(r[h])).join(","))].join("\r\n") + "\r\n";

const workbook = await Workbook.fromCSV(csv, { sheetName: "Ground Truth" });
const inspected = await workbook.inspect({ kind: "table", range: "Ground Truth!A1:I8", include: "values", tableMaxRows: 8, tableMaxCols: 9, maxChars: 6000 });
await fs.writeFile(outputPath, "\uFEFF" + csv, "utf8");

const coverage = Object.fromEntries(files.map(f => [f, rows.filter(r => r.source_file === f).length]));
console.log(inspected.ndjson);
console.log(JSON.stringify({ outputPath, pdfCount: files.length, rowCount: rows.length, coverage }, null, 2));
