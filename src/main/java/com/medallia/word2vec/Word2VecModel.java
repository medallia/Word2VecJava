package com.medallia.word2vec;

import com.google.common.collect.ImmutableList;
import com.google.common.primitives.Doubles;
import com.medallia.word2vec.thrift.Word2VecModelThrift;

import java.util.List;

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
				Doubles.toArray(thrift.getVectors())
			);
	}
	
	/** @return {@link Word2VecTrainerBuilder} for training a model */
	public static Word2VecTrainerBuilder trainer() {
		return new Word2VecTrainerBuilder();
	}
}
