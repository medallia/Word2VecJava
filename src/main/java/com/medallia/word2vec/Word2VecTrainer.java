package com.medallia.word2vec;

import com.google.common.base.Optional;
import com.google.common.base.Predicate;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSortedMultiset;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multisets;
import com.google.common.primitives.Doubles;
import com.medallia.word2vec.util.AC;
import com.medallia.word2vec.util.ProfilingTimer;
import org.apache.commons.logging.Log;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener.Stage;
import com.medallia.word2vec.huffman.HuffmanCoding;
import com.medallia.word2vec.huffman.HuffmanCoding.HuffmanNode;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkConfig;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkTrainer.NeuralNetworkModel;

import java.util.List;
import java.util.Map;

/** Responsible for training a word2vec model */
class Word2VecTrainer {
	private final int minFrequency;
	private final Optional<Multiset<String>> vocab;
	private final NeuralNetworkConfig neuralNetworkConfig;
	
	Word2VecTrainer(
			Integer minFrequency,
			Optional<Multiset<String>> vocab,
			NeuralNetworkConfig neuralNetworkConfig) {
		this.vocab = vocab;
		this.minFrequency = minFrequency;
		this.neuralNetworkConfig = neuralNetworkConfig;
	}

	/** @return {@link Multiset} containing unique tokens and their counts */
	private static Multiset<String> count(Iterable<String> tokens) {
		Multiset<String> counts = HashMultiset.create();
		for (String token : tokens)
			counts.add(token);
		return counts;
	}
	
	/** @return Tokens with their count, sorted by frequency decreasing, then lexicographically ascending */
	private ImmutableMultiset<String> filterAndSort(final Multiset<String> counts) {
		// This isn't terribly efficient, but it is deterministic
		// Unfortunately, Guava's multiset doesn't give us a clean way to order both by count and element
		return Multisets.copyHighestCountFirst(
				ImmutableSortedMultiset.copyOf(
						Multisets.filter(
								counts,
								new Predicate<String>() {
									@Override
									public boolean apply(String s) {
										return counts.count(s) >= minFrequency;
									}
								}
						)
				)
		);
		
	}
	
	/** Train a model using the given data */
	Word2VecModel train(Log log, TrainingProgressListener listener, Iterable<List<String>> sentences) throws InterruptedException {
		try (ProfilingTimer timer = ProfilingTimer.createLoggingSubtasks(log, "Training word2vec")) {
			final Multiset<String> counts;
			
			try (AC ac = timer.start("Acquiring word frequencies")) {
				listener.update(Stage.ACQUIRE_VOCAB, 0.0);
				counts = (vocab.isPresent())
							? vocab.get()
							: count(Iterables.concat(sentences));
			}
			
			final ImmutableMultiset<String> vocab;
			try (AC ac = timer.start("Filtering and sorting vocabulary")) {
				listener.update(Stage.FILTER_SORT_VOCAB, 0.0);
				vocab = filterAndSort(counts);
			}
			
			final Map<String, HuffmanNode> huffmanNodes;
			try (AC task = timer.start("Create Huffman encoding")) {
				huffmanNodes = new HuffmanCoding(vocab, listener).encode();
			}
			
			final NeuralNetworkModel model;
			try (AC task = timer.start("Training model %s", neuralNetworkConfig)) {
				model = neuralNetworkConfig.createTrainer(vocab, huffmanNodes, listener).train(sentences);
			}
			
			return new Word2VecModel(vocab.elementSet(), model.layerSize(), Doubles.concat(model.vectors()));
		}
	}
}
