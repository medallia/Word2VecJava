package com.medallia.word2vec;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.List;

import org.junit.Test;

import com.medallia.word2vec.Searcher.UnknownWordException;

public class Word2VecBinReaderTest {

  @Test
  public void test()
      throws IOException, UnknownWordException {
    URL url = this.getClass().getResource("/com/medallia/word2vec/tokensModel.bin");
    File binFile = new File(url.getFile());
    Word2VecModel binModel = Word2VecModel.fromBinFile(binFile);
    Searcher binSearcher = binModel.forSearch();

    url = this.getClass().getResource("/com/medallia/word2vec/tokensModel.txt");
    File txtFile = new File(url.getFile());
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
      testVector(txtVector, binVector, vocab);
    }
  }

  private void testVector(List<Double> txtVector, List<Double> binVector,
      String vocab) {
    assertEquals(txtVector.size(), binVector.size());
    for (int i = 0; i < txtVector.size(); i++) {
      double txtD = txtVector.get(i);
      double binD = binVector.get(i);
      assertEquals(txtD, binD, 0.0001);
    }
  }
}
