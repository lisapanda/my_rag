"""Prepare Wikipedia HTML without moving inline words out of sentences."""
from bs4 import BeautifulSoup, Comment
import re
from pathlib import Path


def movie_title(raw_html: str, file_name: str) -> str:
    """Find an article title, falling back to the page title or filename."""
    soup = BeautifulSoup(raw_html, "html.parser")
    # Prefer the visible article heading when the saved page includes one.
    heading = soup.select_one("h1")
    if heading and heading.get_text(" ", strip=True):
        return heading.get_text(" ", strip=True)
    # Otherwise use the browser-page title, removing its Wikipedia suffix.
    # If neither HTML title exists, turn a filename like The_Matrix.html into words.
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return re.sub(r"\s*[-–—]\s*Wikipedia$", "", title).strip() or Path(file_name).stem.replace("_", " ")


def is_heading_only(text: str, metadata: dict) -> bool:
    """Reject heading labels without rejecting short factual answers."""
    # Collect the heading labels recorded by the HTML heading splitter.
    # Compare case-insensitively after trimming surrounding whitespace.
    headings = {str(value).strip().casefold() for key, value in metadata.items()
                if key.startswith("header") and value}
    # Ignore blank lines; remaining lines must all be heading labels to be rejected.
    # This keeps short facts such as "$230,000" unless they equal a heading label.
    lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
    return not lines or all(line in headings for line in lines)


def prepare_article_html(raw_html: str) -> str:
    """Clean the article DOM while retaining headings and inline text order."""
    soup = BeautifulSoup(raw_html, "html.parser")
    # Remove invisible editorial comments so they cannot enter embedded text.
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    # decompose() removes both these non-article elements and their contents.
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    # Drop Wikipedia citation superscripts and section-edit controls.
    for tag in soup.select("sup.reference, .mw-editsection"):
        tag.decompose()
    # Prefer Wikipedia's article container, with fallbacks for other saved layouts.
    content = soup.select_one("#mw-content-text")
    if content is None:
        content = soup.body
    if content is None:
        content = soup
    # Keep structural tags (paragraphs, lists, tables, headings) for splitting.
    # Flatten inline wrappers so the splitter sees linked words in sentence order.
    inline_tags = {
        "a", "span", "b", "i", "em", "strong", "small", "sup", "sub",
        "abbr", "cite", "q", "time", "code", "s", "u", "mark", "font",
        "bdi", "bdo", "del", "ins", "wbr",
    }
    # Snapshot the tag list because the loop changes the DOM as it walks it.
    for tag in list(content.find_all(True)):
        if tag.name == "br":
            # Preserve a word boundary where an explicit line break appeared.
            tag.replace_with(" ")
        elif tag.name in inline_tags:
            # unwrap() removes only the wrapper, retaining its text and children.
            # Unlike decompose(), this keeps names inside links in their positions.
            tag.unwrap()
    # Return cleaned HTML, rather than plain text, for the heading splitter.
    return str(content)
