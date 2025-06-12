# src/rag_system/utils/chunking.py
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

@dataclass
class TextChunk:
    content: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: Dict[str, Any] = None

class TextChunker:
    """Simple text chunking with overlap"""
    
    def __init__(
        self, 
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function: callable = len
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text into overlapping chunks"""
        # Clean text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = self.length_function(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = " ".join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_content,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_content),
                    chunk_index=chunk_index
                ))
                chunk_index += 1
                
                # Calculate overlap
                overlap_size = 0
                overlap_chunks = []
                
                # Add sentences from the end until we reach overlap size
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_size += self.length_function(current_chunk[i])
                    overlap_chunks.insert(0, current_chunk[i])
                    if overlap_size >= self.chunk_overlap:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_chunks
                current_length = overlap_size
                chunk_start = chunk_start + len(chunk_content) - overlap_size
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunks.append(TextChunk(
                content=chunk_content,
                start_char=chunk_start,
                end_char=chunk_start + len(chunk_content),
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy or NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

class SemanticChunker:
    """Advanced semantic chunking (placeholder for future implementation)"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        # TODO: Implement semantic chunking based on embedding similarity
        pass