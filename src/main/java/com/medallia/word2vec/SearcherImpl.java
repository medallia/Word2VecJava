package com.medallia.word2vec;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.primitives.Doubles;
import com.medallia.word2vec.util.Pair;

import java.util.Arrays;
import java.util.List;

/** Implementation of {@link Searcher} */
class SearcherImpl implements Searcher {
	private final Word2VecModel model;
	private final ImmutableMap<String, double[]> normalized;

	SearcherImpl(Word2VecModel model) {
		this.model = model;
		ImmutableMap.Builder<String, double[]> result = ImmutableMap.builder();
		for (int i = 0; i < model.vocab.size(); i++) {
			double[] m = Arrays.copyOfRange(model.vectors, i * model.layerSize, (i + 1) * model.layerSize);
			normalize(m);
			result.put(model.vocab.get(i), m);
		}

		normalized = result.build();
	}

	private void normalize(double[] v) {
		double len = 0;
		for (double d : v)
			len += d * d;
		len = Math.sqrt(len);

		for (int i = 0; i < v.length; i++)
			v[i] /= len;
	}

	@Override public List<Match> getMatches(String s, int maxNumMatches) throws UnknownWordException {
		return getMatches(getVector(s), maxNumMatches);
	}

	@Override public double cosineDistance(String s1, String s2) throws UnknownWordException {
		return calculateDistance(getVector(s1), getVector(s2));
	}

	@Override public boolean contains(String word) {
		return normalized.containsKey(word);
	}

	@Override public List<Match> getMatches(final double[] vec, int maxNumMatches) {
		return Match.ORDERING.greatestOf(
				Iterables.transform(model.vocab, new Function<String, Match>() {
					@Override
					public Match apply(String other) {
						double[] otherVec = normalized.get(other);
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
		if (!normalized.containsKey(word))
			throw new UnknownWordException(word);
		return normalized.get(word);
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
