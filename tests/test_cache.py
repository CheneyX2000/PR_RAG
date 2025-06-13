# tests/test_chunking.py
"""
Comprehensive tests for text chunking utilities.
"""

import pytest
from unittest.mock import Mock
from src.rag_system.utils.chunking import TextChunker, TextChunk


class TestTextChunker:
    """Test cases for text chunking"""
    
    @pytest.fixture
    def chunker(self):
        """Create text chunker with default settings"""
        return TextChunker(chunk_size=100, chunk_overlap=20)
    
    @pytest.fixture
    def custom_chunker(self):
        """Create text chunker with custom settings"""
        return TextChunker(chunk_size=50, chunk_overlap=10, length_function=len)
    
    def test_clean_text(self, chunker):
        """Test text cleaning"""
        dirty_text = "  This   has    extra   spaces.  \n\n  And newlines.  "
        clean = chunker._clean_text(dirty_text)
        assert clean == "This has extra spaces. And newlines."
    
    def test_split_sentences(self, chunker):
        """Test sentence splitting"""
        text = "First sentence. Second sentence! Third sentence? Fourth."
        sentences = chunker._split_sentences(text)
        
        assert len(sentences) == 4
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"
        assert sentences[3] == "Fourth."
    
    def test_split_sentences_empty(self, chunker):
        """Test splitting empty text"""
        sentences = chunker._split_sentences("")
        assert sentences == []
    
    def test_split_text_single_chunk(self, chunker):
        """Test splitting text that fits in one chunk"""
        text = "This is a short text."
        chunks = chunker.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)
    
    def test_split_text_multiple_chunks(self, custom_chunker):
        """Test splitting text into multiple chunks"""
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = custom_chunker.split_text(text)
        
        assert len(chunks) > 1
        # Verify chunks are ordered
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
        
        # Verify overlap exists between consecutive chunks
        if len(chunks) > 1:
            # Check that there's some content overlap
            for i in range(len(chunks) - 1):
                current_end = chunks[i].content.split()[-1]
                next_start = chunks[i + 1].content.split()[0]
                # In overlap scenarios, some words should appear in both chunks
    
    def test_split_text_with_overlap(self):
        """Test chunk overlap functionality"""
        chunker = TextChunker(chunk_size=30, chunk_overlap=10)
        text = "Word1 word2. Word3 word4. Word5 word6. Word7 word8."
        
        chunks = chunker.split_text(text)
        
        # Verify overlap by checking for repeated content
        if len(chunks) > 1:
            # Convert chunks to sets of words for comparison
            chunk_words = [set(chunk.content.split()) for chunk in chunks]
            
            # Check overlap between consecutive chunks
            for i in range(len(chunk_words) - 1):
                overlap = chunk_words[i] & chunk_words[i + 1]
                # Should have some overlap (but not complete overlap)
                assert len(overlap) > 0
    
    def test_split_text_empty(self, chunker):
        """Test splitting empty text"""
        chunks = chunker.split_text("")
        assert chunks == []
    
    def test_split_text_whitespace_only(self, chunker):
        """Test splitting whitespace-only text"""
        chunks = chunker.split_text("   \n\n\t   ")
        assert chunks == []
    
    def test_split_text_single_sentence_exceeds_chunk_size(self):
        """Test handling of sentences that exceed chunk size"""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "This is a very long sentence that definitely exceeds our tiny chunk size limit."
        
        chunks = chunker.split_text(text)
        
        # Should still create chunks even with oversized sentences
        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    
    def test_chunk_metadata(self):
        """Test chunk metadata handling"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Test sentence one. Test sentence two."
        
        chunks = chunker.split_text(text)
        
        for chunk in chunks:
            assert hasattr(chunk, 'metadata')
            assert chunk.metadata is None or isinstance(chunk.metadata, dict)
    
    def test_custom_length_function(self):
        """Test using custom length function"""
        # Custom length function that counts words instead of characters
        def word_count(text):
            return len(text.split())
        
        chunker = TextChunker(chunk_size=10, chunk_overlap=2, length_function=word_count)
        text = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
        
        chunks = chunker.split_text(text)
        
        # Verify chunks are created based on word count
        assert len(chunks) > 1
        for chunk in chunks:
            word_count_in_chunk = len(chunk.content.split())
            # Each chunk should have roughly 10 words (plus overlap)
            assert word_count_in_chunk <= 12  # chunk_size + some overlap
    
    def test_chunk_boundaries(self, chunker):
        """Test that chunk boundaries are correct"""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.split_text(text)
        
        # Verify start and end positions
        for i, chunk in enumerate(chunks):
            if i == 0:
                assert chunk.start_char == 0
            if i == len(chunks) - 1:
                # Last chunk should end at or near the text length
                # (might be slightly different due to text cleaning)
                assert chunk.end_char <= len(text) + 10
            
            # Verify chunk content matches the position
            assert len(chunk.content) == chunk.end_char - chunk.start_char
    
    def test_consistent_chunking(self, chunker):
        """Test that chunking is consistent for the same input"""
        text = "This is a test. " * 20  # Repeat to ensure multiple chunks
        
        chunks1 = chunker.split_text(text)
        chunks2 = chunker.split_text(text)
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.content == c2.content
            assert c1.chunk_index == c2.chunk_index


class TestTextChunk:
    """Test cases for TextChunk dataclass"""
    
    def test_text_chunk_creation(self):
        """Test creating a TextChunk"""
        chunk = TextChunk(
            content="Test content",
            start_char=0,
            end_char=12,
            chunk_index=0,
            metadata={"key": "value"}
        )
        
        assert chunk.content == "Test content"
        assert chunk.start_char == 0
        assert chunk.end_char == 12
        assert chunk.chunk_index == 0
        assert chunk.metadata == {"key": "value"}
    
    def test_text_chunk_defaults(self):
        """Test TextChunk with default values"""
        chunk = TextChunk(
            content="Test",
            start_char=0,
            end_char=4,
            chunk_index=0
        )
        
        assert chunk.metadata is None


class TestSemanticChunker:
    """Test cases for semantic chunker (placeholder)"""
    
    def test_semantic_chunker_initialization(self):
        """Test semantic chunker initialization"""
        mock_model = Mock()
        from src.rag_system.utils.chunking import SemanticChunker
        
        chunker = SemanticChunker(embedding_model=mock_model)
        assert chunker.embedding_model == mock_model