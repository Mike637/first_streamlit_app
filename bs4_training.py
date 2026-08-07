from pathlib import Path
from bs4 import (BeautifulSoup,
                 NavigableString)
from bs4.exceptions import FeatureNotFound
from bs4.element import Tag
from typing import (Union,
                    Optional)
import logging
from typing import List
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

CURRENT_DIR = Path(__file__).parent
HELP_PATH = CURRENT_DIR / 'help'
HTML_FILE_PATH = Path(r'D:\first_streamlit_app\help\step_by_step_tutorials\step_by_step_tutorials.htm')
SAVE_PATH = CURRENT_DIR / 'text_files'


# SAVE_PATH.mkdir()

class HtmlParser:
    html_path: Path

    def __init__(self, html_path: Union[Path, str]) -> None:
        self.html_path = Path(html_path).resolve(strict=False)

    def read_html(self) -> Optional[str]:
        if self.html_path.suffix not in ('.html', '.htm'):
            logging.error(f'File in {self.html_path} must have .html or .htm extension!')
            return
        try:
            with open(self.html_path, 'r', encoding='utf-8') as html_file:
                return html_file.read()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            logging.error(f'File is not found in {self.html_path}!')
            return
        except PermissionError:
            logging.error(f'Permission to {self.html_path} denied!')
            return
        except OSError as e:
            logging.exception(f'OS error while opening {self.html_path}:{e}')
            return

    def create_html_parser(self) -> Optional[BeautifulSoup]:
        soap = None
        html = self.read_html()
        if html is None:
            return
        try:
            soap = BeautifulSoup(html, 'html.parser')
        except FeatureNotFound as e:
            logging.error(f'{e}')
            return
        return soap

    def main(self):
        def get_text_from_row(tr, header_table_lines):
            text = []
            table_lines = [line.get_text(strip=True) or 'пустая ячейка таблицы' for line in tr.find_all(['td', 'th'])]
            for table_line, header_table_line in zip(table_lines, header_table_lines):
                table_line = table_line.replace('\n', '').replace('\t', '')
                text.append(f"{header_table_line}:{table_line}")
            return text

        table_text = []
        soap = self.create_html_parser()

        body = soap.find('body')

        if not body:
            return
        title = soap.find('title')

        title_text = '[TITLE]' + title.get_text() + '[/TITLE]' if title else ''
        delete_tags = ['script', 'noscript', 'link', 'footer', 'style', 'div#header', 'form']

        for tag in body.select(', '.join(delete_tags)):
            tag.decompose()

        for tag in body.find_all(['table']):
            trs = tag.find_all(['tr'])
            table_header_rows = trs[0]
            rows = trs[1:]
            header_lines = [line.get_text(strip=True) or 'заголовок отсутствует в ячейке' for line in
                            table_header_rows.find_all(['td', 'th'])]
            for row in rows:
                table_text.extend(get_text_from_row(row, header_lines))
            tag.replace_with('\n[TABLE]\n ' + "\n".join(line for line in table_text) + ' \n[/TABLE]\n')
        for tag in body.find_all(["ul"]):
            for list in tag.find_all(["li"]):
                list.string = list.get_text().replace('\n', '').replace('\t', '')
            tag.string = '[LIST]' + tag.get_text() + '[/LIST]'
        for tag in body.find_all(["blockquote", "pre"]):
            for paragraph in tag.select('p.command'):
                paragraph.string = paragraph.get_text()
            tag.string = f'[COMMANDS] {tag.get_text()}\n[/COMMANDS]'
        for tag in body.find_all(['img']):
            src = tag["src"]
            tag.replace_with(f'[IMAGE] {src} [/IMAGE]')

        for tag in body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7']):
            heading_text = tag.get_text().replace("\n", "")
            tag.string = f'[HEADING_{tag.name}] {heading_text} [/HEADING_{tag.name}]\n'
        for tag in body.find_all(['p']):
            parargraph_text = tag.get_text(" ", strip=True).replace('\n', '')
            tag.string = '[PARAGRAPH]' + parargraph_text + '[/PARAGRAPH]'

        lines = [line for line in body.get_text(strip=False).split('\n') if line]
        return f'{title_text}\n' + '\n'.join(line for line in lines).replace('\t', '').replace('\xa0', ' ')


def get_html_paths(folder_path: str) -> List[str]:
    html_files_list = []
    for path, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(('.html', '.htm')):
                html_files_list.append(os.path.join(path, f))
    return html_files_list


def save_txt_file(args):
    path, text = args
    with path.open('w', encoding='utf-8') as file:
        file.write(f'{path}\n')
        file.write('\n')
        file.write(text)


htmls = get_html_paths(HELP_PATH)


def main():
    tasks = []
    for path in htmls:
        parser = HtmlParser(path)
        text = parser.main()
        if not text:
            continue
        name = Path(path).name
        save_path = SAVE_PATH / name
        tasks.append((save_path, text))
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(save_txt_file, tasks), desc=f'Saving file...', total=len(tasks)))


def collect_all_documentation():
    text = ""
    for path in htmls:
        parser = HtmlParser(path)
        doc = parser.main()
        if not doc:
            continue
        text += f'{doc}\n'
    return text


if __name__ == '__main__':
    main()
    #res = collect_all_documentation()
    #print(res)
