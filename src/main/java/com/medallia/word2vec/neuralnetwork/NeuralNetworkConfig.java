package com.medallia.word2vec.neuralnetwork;

import com.google.common.collect.ImmutableMultiset;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.medallia.word2vec.huffman.HuffmanCoding.HuffmanNode;

import java.util.Map;

/** Fixed configuration for training the neural network */
public class NeuralNetworkConfig {
	final int numThreads;
	final int iterations;
	final NeuralNetworkType type;
	final int layerSize;
	final int windowSize;
	final int negativeSamples;
	final boolean useHierarchicalSoftmax;
	
	final double initialLearningRate;
	final double downSampleRate;
	
	/** Constructor */
	public NeuralNetworkConfig(
			NeuralNetworkType type,
			int numThreads,
			int iterations,
			int layerSize,
			int windowSize,
			int negativeSamples,
			double downSampleRate,
			double initialLearningRate,
			boolean useHierarchicalSoftmax) {
		this.type = type;
		this.iterations = iterations;
		this.numThreads = numThreads;
		this.layerSize = layerSize;
		this.windowSize = windowSize;
		this.negativeSamples = negativeSamples;
		this.useHierarchicalSoftmax = useHierarchicalSoftmax;
		this.initialLearningRate = initialLearningRate;
		this.downSampleRate = downSampleRate;
	}

	/** @return {@link NeuralNetworkTrainer} */
	public NeuralNetworkTrainer createTrainer(ImmutableMultiset<String> vocab, Map<String, HuffmanNode> huffmanNodes, TrainingProgressListener listener) {
		return type.createTrainer(this, vocab, huffmanNodes, listener);
	}
	
	@Override public String toString() {
		return String.format("%s with %s threads, %s iterations[%s layer size, %s window, %s hierarchical softmax, %s negative samples, %s initial learning rate, %s down sample rate]",
				type.name(),
				numThreads,
				iterations,
				layerSize,
				windowSize,
				useHierarchicalSoftmax ? "using" : "not using",
				negativeSamples, 
				initialLearningRate,
				downSampleRate
			);
	}
}
