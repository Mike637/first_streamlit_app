from langchain.text_splitter import RecursiveCharacterTextSplitter
from htms_parsing import collect_all_documentation
from typing import (List,
                    Final
                    )
import re
from langchain_core.documents import Document

TITLE_SEPARATOR: Final[List[str]] = ['[TITLE]']
HEADING_SEPARATORS: Final[List[str]] = [
    '[HEADING_h1]',
    '[HEADING_h2]',
    '[HEADING_h3]',
    '[HEADING_h4]',
    '[HEADING_h5]',
    '[HEADING_h6]',
    '[HEADING_h7]'
]
SEPARATORS: Final[List[str]] = [
    '[PARAGRAPH]',
    '[TABLE]',
    '[COMMANDS]',
    '[LIST]'
]
FINAL_SEPARATORS: Final[List[str]] = [
    "\n\n",
    "\n",
    ". ",
    " ",
    ""
]
DOCUMENTATION_DATA: str = collect_all_documentation()


def split_by_splitter(text: str, separators: List[str], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(separators=separators,
                                              chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    split_text = splitter.split_text(text)
    return split_text


def split_by_tags(text: str, tags: List[str]) -> List[str]:
    pattern = "(?=" + "|".join(map(re.escape, tags)) + ")"
    return [x for x in re.split(pattern, text) if x.strip()]


chunked_docs_by_tags = []
for doc in DOCUMENTATION_DATA:
    path = doc.metadata.get('source', 'Not Found')
    text = doc.page_content
    for section in split_by_tags(text, HEADING_SEPARATORS):
        for chunk in split_by_tags(section, SEPARATORS):
            chunked_docs_by_tags.append(Document(
                page_content=chunk,
                metadata={'source': path}
            ))

'''
splitted_text_by_tags = []
splitted_text_by_splitter = []
for text in title_split_text:
    for line in split_by_tags(text, HEADING_SEPARATORS):
        for row in split_by_tags(line, SEPARATORS):
            splitted_text_by_tags.append(row)
for text in splitted_text_by_tags:
    splitted_text = split_by_splitter(text, FINAL_SEPARATORS)
    splitted_text_by_splitter.extend(splitted_text)

print(splitted_text_by_splitter[56])
'''
