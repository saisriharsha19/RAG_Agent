import os
import json
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_documents(self, directory_path: str) -> List[LangchainDocument]:
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"Directory {directory_path} does not exist")
            return documents
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    content = self._extract_content(file_path)
                    if content:
                        doc = LangchainDocument(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                                "file_type": file_path.suffix.lower()
                            }
                        )
                        documents.append(doc)
                        print(f"Loaded: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        
        return documents
    
    def _extract_content(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_pdf(file_path)
        elif extension == '.docx':
            return self._extract_docx(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self._extract_excel(file_path)
        elif extension == '.txt':
            return self._extract_text(file_path)
        elif extension == '.json':
            return self._extract_json(file_path)
        else:
            print(f"Unsupported file type: {extension}")
            return ""
    
    def _extract_pdf(self, file_path: Path) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_docx(self, file_path: Path) -> str:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_excel(self, file_path: Path) -> str:
        df = pd.read_excel(file_path, sheet_name=None)
        text = ""
        for sheet_name, sheet_df in df.items():
            text += f"Sheet: {sheet_name}\n"
            text += sheet_df.to_string(index=False) + "\n\n"
        return text
    
    def _extract_text(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _extract_json(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return json.dumps(data, indent=2)
    
    # Add this method to your DocumentProcessor class

    def load_single_document(self, file_path: str):
        """Load a single document by file path"""
        try:
            from pathlib import Path
            
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Determine file type and process accordingly
            if file_path.suffix.lower() == '.pdf':
                return self._load_pdf(file_path)
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                return self._load_docx(file_path)
            elif file_path.suffix.lower() == '.txt':
                return self._load_txt(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return self._load_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                return self._load_json(file_path)
            elif file_path.suffix.lower() in ['.csv']:
                return self._load_csv(file_path)
            elif file_path.suffix.lower() == '.md':
                return self._load_markdown(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None

    def _load_txt(self, file_path):
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'file_type': 'txt',
                    'file_path': str(file_path)
                }
            }
        except Exception as e:
            logger.error(f"Error loading TXT file {file_path}: {e}")
            return None

    def _load_markdown(self, file_path):
        """Load markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'file_type': 'markdown',
                    'file_path': str(file_path)
                }
            }
        except Exception as e:
            logger.error(f"Error loading Markdown file {file_path}: {e}")
            return None

    def _load_csv(self, file_path):
        """Load CSV file"""
        try:
            import pandas as pd
            
            df = pd.read_csv(file_path)
            # Convert DataFrame to readable text format
            content = f"CSV Data from {file_path.name}:\n\n"
            content += f"Columns: {', '.join(df.columns)}\n"
            content += f"Rows: {len(df)}\n\n"
            content += df.to_string(index=False)
            
            return {
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'file_type': 'csv',
                    'file_path': str(file_path),
                    'rows': len(df),
                    'columns': list(df.columns)
                }
            }
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return None

    def _load_json(self, file_path):
        """Load JSON file"""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable text format
            content = f"JSON Data from {file_path.name}:\n\n"
            content += json.dumps(data, indent=2)
            
            return {
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'file_type': 'json',
                    'file_path': str(file_path)
                }
            }
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return None

    def _load_excel(self, file_path):
        """Load Excel file"""
        try:
            import pandas as pd
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            content = f"Excel Data from {file_path.name}:\n\n"
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content += f"Sheet: {sheet_name}\n"
                content += f"Columns: {', '.join(df.columns)}\n"
                content += f"Rows: {len(df)}\n\n"
                content += df.to_string(index=False)
                content += "\n" + "="*50 + "\n\n"
            
            return {
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'file_type': 'excel',
                    'file_path': str(file_path),
                    'sheets': excel_file.sheet_names
                }
            }
        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {e}")
            return None

    def _load_pdf(self, file_path):
        """Load PDF file"""
        try:
            import PyPDF2
            
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    content += f"--- Page {page_num + 1} ---\n"
                    content += page.extract_text()
                    content += "\n"
            
            return {
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'file_type': 'pdf',
                    'file_path': str(file_path),
                    'pages': len(pdf_reader.pages)
                }
            }
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return None

    def _load_docx(self, file_path):
        """Load DOCX file"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            content = ""
            
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            
            return {
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'file_type': 'docx',
                    'file_path': str(file_path),
                    'paragraphs': len(doc.paragraphs)
                }
            }
        except Exception as e:
            logger.error(f"Error loading DOCX file {file_path}: {e}")
            return None

    # Also add this method to get document statistics
    def get_document_stats(self, documents):
        """Get statistics about loaded documents"""
        if not documents:
            return {}
        
        stats = {
            'total_documents': len(documents),
            'file_types': {},
            'total_content_length': 0,
            'average_content_length': 0
        }
        
        for doc in documents:
            # Count file types
            file_type = doc.get('metadata', {}).get('file_type', 'unknown')
            stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            # Calculate content length
            content_length = len(doc.get('content', ''))
            stats['total_content_length'] += content_length
        
        if stats['total_documents'] > 0:
            stats['average_content_length'] = stats['total_content_length'] / stats['total_documents']
        
        return stats
    
    def split_documents(self, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        return self.text_splitter.split_documents(documents)