package com.medallia.word2vec;

import com.medallia.word2vec.thrift.Word2VecModelThrift;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;

/**
 * Represents a word2vec model where all the vectors are normalized to unit length.
 */
public class NormalizedWord2VecModel extends Word2VecModel {
	private NormalizedWord2VecModel(Iterable<String> vocab, int layerSize, final DoubleBuffer[] vectors) {
		super(vocab, layerSize, vectors);
		normalize();
	}

	private NormalizedWord2VecModel(Iterable<String> vocab, int layerSize, double[] vectors) {
		super(vocab, layerSize, vectors);
		normalize();
	}

	public static NormalizedWord2VecModel fromWord2VecModel(Word2VecModel model) {
		DoubleBuffer[] newVectors = new DoubleBuffer[model.vectors.length];
		for (int i = 0; i < newVectors.length; i++) {
			newVectors[i] = model.vectors[i].duplicate();
		}
		return new NormalizedWord2VecModel(model.vocab, model.layerSize, newVectors);
	}

	/** @return {@link NormalizedWord2VecModel} created from a thrift representation */
	public static NormalizedWord2VecModel fromThrift(final Word2VecModelThrift thrift) {
		return fromWord2VecModel(Word2VecModel.fromThrift(thrift));
	}

	public static NormalizedWord2VecModel fromBinFile(final File file) throws IOException {
		return fromWord2VecModel(Word2VecModel.fromBinFile(file));
	}

	/** Normalizes the vectors in this model */
	private void normalize() {
		for(int i = 0; i < vectors.length; ++i) {
			DoubleBuffer buffer = vectors[i];
			for(int j = 0; j < Math.min(vectorsPerBuffer, buffer.limit() / layerSize); ++j) {
				double len = 0;
				for(int k = j * layerSize; k < (j + 1) * layerSize; ++k)
					len += buffer.get(k) * buffer.get(k);
				len = Math.sqrt(len);

				for(int k = j * layerSize; k < (j + 1) * layerSize; ++k)
					buffer.put(k, buffer.get(k) / len);
			}
		}
	}
}
