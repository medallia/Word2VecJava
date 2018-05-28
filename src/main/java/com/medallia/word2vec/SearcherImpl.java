package com.medallia.word2vec;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.primitives.Doubles;
import com.medallia.word2vec.util.Pair;

import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.List;

/** Implementation of {@link Searcher} */
class SearcherImpl implements Searcher {
  private final NormalizedWord2VecModel model;
	private final ImmutableMap<String, Long> word2vectorOffset;
	private final int bufferSize;

	SearcherImpl(final NormalizedWord2VecModel model) {
		this.bufferSize = model.layerSize * model.vectorsPerBuffer;
		long maxIndex = ((long) model.vocab.size() - 1) * model.layerSize;
		// We use the vocab index divided by the buffer size as an array index, so it must fit into an int.
		Preconditions.checkArgument(
				maxIndex / bufferSize < Integer.MAX_VALUE,
				"vocabulary and / or vector size is too large to calculate indexes for"
		);
		this.model = model;

		final ImmutableMap.Builder<String, Long> result = ImmutableMap.builder();
		for (int i = 0; i < model.vocab.size(); i++) {
			result.put(model.vocab.get(i), ((long) i) * model.layerSize);
		}

		word2vectorOffset = result.build();
	}

  SearcherImpl(final Word2VecModel model) {
	this(NormalizedWord2VecModel.fromWord2VecModel(model));
  }

  @Override public List<Match> getMatches(String s, int maxNumMatches) throws UnknownWordException {
	return getMatches(getVector(s), maxNumMatches);
  }

  @Override public double cosineDistance(String s1, String s2) throws UnknownWordException {
	return calculateDistance(getVector(s1), getVector(s2));
  }

  @Override public boolean contains(String word) {
	return word2vectorOffset.containsKey(word);
  }

  @Override public List<Match> getMatches(final double[] vec, int maxNumMatches) {
	return Match.ORDERING.greatestOf(
		Iterables.transform(model.vocab, new Function<String, Match>() {
		  @Override
		  public Match apply(String other) {
			double[] otherVec = getVectorOrNull(other);
			double d = calculateDistance(otherVec, vec);
			return new MatchImpl(other, d);
		  }
		}),
		maxNumMatches
	);
  }

  private double calculateDistance(double[] otherVec, double[] vec) {
	double d = 0;
	for (int a = 0; a < model.layerSize; a++)
	  d += vec[a] * otherVec[a];
	return d;
  }

  @Override public ImmutableList<Double> getRawVector(String word) throws UnknownWordException {
	return ImmutableList.copyOf(Doubles.asList(getVector(word)));
  }

  /**
   * @return Vector for the given word
   * @throws UnknownWordException If word is not in the model's vocabulary
   */
  private double[] getVector(String word) throws UnknownWordException {
	final double[] result = getVectorOrNull(word);
	if(result == null)
	  throw new UnknownWordException(word);

	return result;
  }

	private double[] getVectorOrNull(final String word) {
		final Long index = word2vectorOffset.get(word);
		if(index == null)
			return null;

		final DoubleBuffer vectors = model.vectors[(int) (index / bufferSize)].duplicate();
		double[] result = new double[model.layerSize];
		vectors.position((int) (index % bufferSize));
		vectors.get(result);
		return result;
	}

  /** @return Vector difference from v1 to v2 */
  private double[] getDifference(double[] v1, double[] v2) {
	double[] diff = new double[model.layerSize];
	for (int i = 0; i < model.layerSize; i++)
	  diff[i] = v1[i] - v2[i];
	return diff;
  }

  @Override public SemanticDifference similarity(String s1, String s2) throws UnknownWordException {
	double[] v1 = getVector(s1);
	double[] v2 = getVector(s2);
	final double[] diff = getDifference(v1, v2);

	return new SemanticDifference() {
	  @Override public List<Match> getMatches(String word, int maxMatches) throws UnknownWordException {
		double[] target = getDifference(getVector(word), diff);
		return SearcherImpl.this.getMatches(target, maxMatches);
	  }
	};
  }

  /** Implementation of {@link Match} */
  private static class MatchImpl extends Pair<String, Double> implements Match {
	private MatchImpl(String first, Double second) {
	  super(first, second);
	}

	@Override public String match() {
	  return first;
	}

	@Override public double distance() {
	  return second;
	}

	@Override public String toString() {
	  return String.format("%s [%s]", first, second);
	}
  }
}
