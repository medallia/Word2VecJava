package com.medallia.word2vec;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteOrder;
import java.util.List;

import org.apache.commons.io.input.SwappedDataInputStream;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.medallia.word2vec.thrift.Word2VecModelThrift;
import com.medallia.word2vec.util.Common;

/**
 * Represents the Word2Vec model, containing vectors for each word
 * <p>
 * Instances of this class are obtained via:
 * <ul>
 * <li> {@link #trainer()}
 * <li> {@link #fromThrift(Word2VecModelThrift)}
 * </ul>
 * 
 * @see {@link #forSearch()}
 */
public class Word2VecModel {
	final List<String> vocab;
	final int layerSize;
	final double[] vectors;
	
	Word2VecModel(Iterable<String> vocab, int layerSize, double[] vectors) {
		this.vocab = ImmutableList.copyOf(vocab);
		this.layerSize = layerSize;
		this.vectors = vectors;
	}
	
	/** @return Vocabulary */
	public Iterable<String> getVocab() {
		return vocab;
	}

	/** @return {@link Searcher} for searching */
	public Searcher forSearch() {
		return new SearcherImpl(this);
	}
	
	/** @return Serializable thrift representation */
	public Word2VecModelThrift toThrift() {
		return new Word2VecModelThrift()
			.setVocab(vocab)
			.setLayerSize(layerSize)
			.setVectors(Doubles.asList(vectors));
	}
	
	/** @return {@link Word2VecModel} created from a thrift representation */
	public static Word2VecModel fromThrift(Word2VecModelThrift thrift) {
		return new Word2VecModel(
				thrift.getVocab(),
				thrift.getLayerSize(),
				Doubles.toArray(thrift.getVectors()));
	}

	/**
	 * @return {@link Word2VecModel} read from a file in the text output format of the Word2Vec C
	 * open source project.
	 */
	public static Word2VecModel fromTextFile(File file) throws IOException {
		List<String> lines = Common.readToList(file);
		return fromTextFile(file.getAbsolutePath(), lines);
	}

	/**
   * Forwards to {@link #fromBinFile(File, ByteOrder)} with the default 
   * ByteOrder.LITTLE_ENDIAN
   */
  public static Word2VecModel fromBinFile(File file)
      throws IOException {
    return fromBinFile(file, ByteOrder.LITTLE_ENDIAN);
  }

  /**
   * @return {@link Word2VecModel} created from the binary representation output
   * by the open source C version of word2vec using the given byte order.
   */
  public static Word2VecModel fromBinFile(File file, ByteOrder byteOrder)
      throws IOException {

    try (FileInputStream fis = new FileInputStream(file);) {
      DataInput in = (byteOrder == ByteOrder.BIG_ENDIAN) ?
          new DataInputStream(fis) : new SwappedDataInputStream(fis);

      StringBuilder sb = new StringBuilder();
      char c = (char) in.readByte();
      while (c != '\n') {
        sb.append(c);
        c = (char) in.readByte();
      }
      String firstLine = sb.toString();
      int index = firstLine.indexOf(' ');
      Preconditions.checkState(index != -1,
          "Expected a space in the first line of file '%s': '%s'",
          file.getAbsolutePath(), firstLine);

      int vocabSize = Integer.parseInt(firstLine.substring(0, index));
      int layerSize = Integer.parseInt(firstLine.substring(index + 1));

      List<String> vocabs = Lists.newArrayList();
      List<Double> vectors = Lists.newArrayList();

      for (int lineno = 0; lineno < vocabSize; lineno++) {
        sb = new StringBuilder();
        c = (char) in.readByte();
        while (c != ' ') {
          // ignore newlines in front of words (some binary files have newline,
          // some don't)
          if (c != '\n') {
            sb.append(c);
          }
          c = (char) in.readByte();
        }
        vocabs.add(sb.toString());

        for (int i = 0; i < layerSize; i++) {
          vectors.add((double) in.readFloat());
        }
      }

      return fromThrift(new Word2VecModelThrift()
          .setLayerSize(layerSize)
          .setVocab(vocabs)
          .setVectors(vectors));
    }
  }

  /**
	 * @return {@link Word2VecModel} from the lines of the file in the text output format of the
	 * Word2Vec C open source project.
	 */
	@VisibleForTesting
	static Word2VecModel fromTextFile(String filename, List<String> lines) throws IOException {
		List<String> vocab = Lists.newArrayList();
		List<Double> vectors = Lists.newArrayList();
		int vocabSize = Integer.parseInt(lines.get(0).split(" ")[0]);
		int layerSize = Integer.parseInt(lines.get(0).split(" ")[1]);

		Preconditions.checkArgument(
				vocabSize == lines.size() - 1,
				"For file '%s', vocab size is %s, but there are %s word vectors in the file",
				filename,
				vocabSize,
				lines.size() - 1
			);

		for (int n = 1; n < lines.size(); n++) {
			String[] values = lines.get(n).split(" ");
			vocab.add(values[0]);

			// Sanity check
			Preconditions.checkArgument(
					layerSize == values.length - 1,
					"For file '%s', on line %s, layer size is %s, but found %s values in the word vector",
					filename,
					n,
					layerSize,
					values.length - 1
				);

			for (int d = 1; d < values.length; d++) {
				vectors.add(Double.parseDouble(values[d]));
			}
		}

		Word2VecModelThrift thrift = new Word2VecModelThrift()
				.setLayerSize(layerSize)
				.setVocab(vocab)
				.setVectors(vectors);
		return fromThrift(thrift);
	}
	
	/** @return {@link Word2VecTrainerBuilder} for training a model */
	public static Word2VecTrainerBuilder trainer() {
		return new Word2VecTrainerBuilder();
	}
}
