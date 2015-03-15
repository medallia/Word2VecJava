package com.medallia.word2vec;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.junit.Test;

import com.medallia.word2vec.Searcher.UnknownWordException;
import com.medallia.word2vec.util.Common;

/**
 * Tests converting the binary models into
 * {@link com.medallia.word2vec.Word2VecModel}s.
 * 
 * @see com.medallia.word2vec.Word2VecModel#fromBinFile(File)
 * @see com.medallia.word2vec.Word2VecModel#fromBinFile(File,
 *      java.nio.ByteOrder)
 */
public class Word2VecBinReaderTest {

  /**
   * Tests that the Word2VecModels created from a binary and text
   * representations are equivalent
   */
  @Test
  public void test()
      throws IOException, UnknownWordException {
    File binFile = Common.getResourceAsFile(
        this.getClass(),
        "/com/medallia/word2vec/tokensModel.bin");
    Word2VecModel binModel = Word2VecModel.fromBinFile(binFile);
    Searcher binSearcher = binModel.forSearch();

    File txtFile = Common.getResourceAsFile(
        this.getClass(),
        "/com/medallia/word2vec/tokensModel.txt");
    Word2VecModel txtModel = Word2VecModel.fromTextFile(txtFile);
    Searcher txtSearcher = txtModel.forSearch();

    // test vocab
    for (String vocab : txtModel.getVocab()) {
      assertTrue(binSearcher.contains(vocab));
    }
    for (String vocab : binModel.getVocab()) {
      assertTrue(txtSearcher.contains(vocab));
    }
    // test vector
    for (String vocab : txtModel.getVocab()) {
      List<Double> txtVector = txtSearcher.getRawVector(vocab);
      List<Double> binVector = binSearcher.getRawVector(vocab);
      assertVectorsEqual(txtVector, binVector);
    }
  }

  private void assertVectorsEqual(List<Double> txtVector,
      List<Double> binVector) {
    assertEquals(txtVector.size(), binVector.size());
    for (int i = 0; i < txtVector.size(); i++) {
      double txtD = txtVector.get(i);
      double binD = binVector.get(i);
      assertEquals(txtD, binD, 0.0001);
    }
  }
}
