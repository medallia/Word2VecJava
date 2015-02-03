package com.medallia.word2vec.neuralnetwork;

import com.google.common.collect.Iterables;
import com.google.common.collect.Multiset;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener.Stage;
import com.medallia.word2vec.huffman.HuffmanCoding.HuffmanNode;
import com.medallia.word2vec.util.CallableVoid;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/** Parent class for training word2vec's neural network */
public abstract class NeuralNetworkTrainer {
	/** Sentences longer than this are broken into multiple chunks */
	private static final int MAX_SENTENCE_LENGTH = 1_000;
	
	/** Boundary for maximum exponent allowed */
	static final int MAX_EXP = 6;
	
	/** Size of the pre-cached exponent table */
	static final int EXP_TABLE_SIZE = 1_000;
	static final double[] EXP_TABLE = new double[EXP_TABLE_SIZE];
	static {
		for (int i = 0; i < EXP_TABLE_SIZE; i++) {
			// Precompute the exp() table
			EXP_TABLE[i] = Math.exp((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
			// Precompute f(x) = x / (x + 1)
			EXP_TABLE[i] /= EXP_TABLE[i] + 1;
		}
	}
	
	private static final int TABLE_SIZE = (int)1e8;
	
	private final TrainingProgressListener listener;
	
	final NeuralNetworkConfig config;
	final Map<String, HuffmanNode> huffmanNodes;
	private final int vocabSize;
	final int layer1_size;
	final int window;
	/** 
	 * In the C version, this includes the </s> token that replaces a newline character
	 */
	int numTrainedTokens;
	
	/* The following includes shared state that is updated per worker thread */
	
	/** 
	 * To be precise, this is the number of words in the training data that exist in the vocabulary
	 * which have been processed so far.  It includes words that are discarded from sampling.
	 * Note that each word is processed once per iteration.
	 */
	protected final AtomicInteger actualWordCount;
	/** Learning rate, affects how fast values in the layers get updated */
	volatile double alpha;
	/** 
	 * This contains the outer layers of the neural network
	 * First dimension is the vocab, second is the layer
	 */
	final double[][] syn0;
	/** This contains hidden layers of the neural network */
	final double[][] syn1;
	/** This is used for negative sampling */
	private final double[][] syn1neg;
	/** Used for negative sampling */
	private final int[] table;
	long startNano;
	
	NeuralNetworkTrainer(NeuralNetworkConfig config, Multiset<String> vocab, Map<String, HuffmanNode> huffmanNodes, TrainingProgressListener listener) {
		this.config = config;
		this.huffmanNodes = huffmanNodes;
		this.listener = listener;
		this.vocabSize = huffmanNodes.size();
		this.numTrainedTokens = vocab.size();
		this.layer1_size = config.layerSize;
		this.window = config.windowSize;
		
		this.actualWordCount = new AtomicInteger();
		this.alpha = config.initialLearningRate;
		
		this.syn0 = new double[vocabSize][layer1_size];
		this.syn1 = new double[vocabSize][layer1_size];
		this.syn1neg = new double[vocabSize][layer1_size];
		this.table = new int[TABLE_SIZE];
		
		initializeSyn0();
		initializeUnigramTable();
	}
	
	private void initializeUnigramTable() {
		long trainWordsPow = 0;
		double power = 0.75;
		
		for (HuffmanNode node : huffmanNodes.values()) {
			trainWordsPow += Math.pow(node.count, power);
		}
		
		Iterator<HuffmanNode> nodeIter = huffmanNodes.values().iterator();
		HuffmanNode last = nodeIter.next();
		double d1 = Math.pow(last.count, power) / trainWordsPow;
		int i = 0;
		for (int a = 0; a < TABLE_SIZE; a++) {
			table[a] = i;
			if (a / (double)TABLE_SIZE > d1) {
				i++;
				HuffmanNode next = nodeIter.hasNext()
						? nodeIter.next()
						: last;
				
				d1 += Math.pow(next.count, power) / trainWordsPow;
				
				last = next;
			}
		}
	}

	private void initializeSyn0() {
		long nextRandom = 1;
		for (int a = 0; a < huffmanNodes.size(); a++) {
			// Consume a random for fun
			// Actually we do this to use up the injected </s> token
			nextRandom = incrementRandom(nextRandom);
			for (int b = 0; b < layer1_size; b++) {
				nextRandom = incrementRandom(nextRandom);
				syn0[a][b] = (((nextRandom & 0xFFFF) / (double)65_536) - 0.5) / layer1_size;
			}
		}
	}
	
	/** @return Next random value to use */
	static long incrementRandom(long r) {
		return r * 25_214_903_917L + 11;
	}

	/** Represents a neural network model */
	public interface NeuralNetworkModel {
		/** Size of the layers */
		int layerSize();
		/** Resulting vectors */
		double[][] vectors();
	}
	
	/** @return Trained NN model */
	public NeuralNetworkModel train(Iterable<List<String>> sentences) throws InterruptedException {
		ListeningExecutorService ex = MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(config.numThreads));
		
		int numSentences = Iterables.size(sentences);
		numTrainedTokens += numSentences;
		
		// Partition the sentences evenly amongst the threads
		Iterable<List<List<String>>> partitioned = Iterables.partition(sentences, numSentences / config.numThreads + 1);
		
		try {
			listener.update(Stage.TRAIN_NEURAL_NETWORK, 0.0);
			for (int iter = config.iterations; iter > 0; iter--) {
				List<CallableVoid> tasks = new ArrayList<>();
				int i = 0;
				for (final List<List<String>> batch : partitioned) {
					tasks.add(createWorker(i, iter, batch));
					i++;
				}
				
				List<ListenableFuture<?>> futures = new ArrayList<>(tasks.size());
				for (CallableVoid task : tasks)
					futures.add(ex.submit(task));
				try {
					Futures.allAsList(futures).get();
				} catch (ExecutionException e) {
					throw new IllegalStateException("Error training neural network", e.getCause());
				}
			}
			ex.shutdown();
		} finally {
			ex.shutdownNow();
		}
		
		return new NeuralNetworkModel() {
			@Override public int layerSize() {
				return config.layerSize;
			}
			
			@Override public double[][] vectors() {
				return syn0;
			}
		};
	}
	
	/** @return {@link Worker} to process the given sentences */
	abstract Worker createWorker(int randomSeed, int iter, Iterable<List<String>> batch);
	
	/** Worker thread that updates the neural network model */
	abstract class Worker extends CallableVoid {
		private static final int LEARNING_RATE_UPDATE_FREQUENCY = 10_000;
		
		long nextRandom;
		final int iter;
		final Iterable<List<String>> batch;
		
		/** 
		 * The number of words observed in the training data for this worker that exist
		 * in the vocabulary.  It includes words that are discarded from sampling.
		 */
		int wordCount;
		/** Value of wordCount the last time alpha was updated */
		int lastWordCount;
		
		final double[] neu1 = new double[layer1_size];
		final double[] neu1e = new double[layer1_size];
		
		Worker(int randomSeed, int iter, Iterable<List<String>> batch) {
			this.nextRandom = randomSeed;
			this.iter = iter;
			this.batch = batch;
		}
		
		@Override public void run() throws InterruptedException {
			for (List<String> sentence : batch) {
				List<String> filteredSentence = new ArrayList<>(sentence.size());
				for (String s : sentence) {
					if (!huffmanNodes.containsKey(s))
						continue;
							
					wordCount++;
					if (config.downSampleRate > 0) {
						HuffmanNode huffmanNode = huffmanNodes.get(s);
						double random = (Math.sqrt(huffmanNode.count / (config.downSampleRate * numTrainedTokens)) + 1)
								* (config.downSampleRate * numTrainedTokens) / huffmanNode.count;
						nextRandom = incrementRandom(nextRandom);
						if (random < (nextRandom & 0xFFFF) / (double)65_536) {
							continue;
						}
					}
					
					filteredSentence.add(s);
				}
				
				// Increment word count one extra for the injected </s> token
				// Turns out if you don't do this, the produced word vectors aren't as tasty
				wordCount++;
				
				Iterable<List<String>> partitioned = Iterables.partition(filteredSentence, MAX_SENTENCE_LENGTH);
				for (List<String> chunked : partitioned) {
					if (Thread.currentThread().isInterrupted())
						throw new InterruptedException("Interrupted while training word2vec model");
					
					if (wordCount - lastWordCount > LEARNING_RATE_UPDATE_FREQUENCY) {
						updateAlpha(iter);
					}
					trainSentence(chunked);
				}
			}
			
			actualWordCount.addAndGet(wordCount - lastWordCount);
		}
		
		/** 
		 * Degrades the learning rate (alpha) steadily towards 0
		 * @param iter Only used for debugging
		 */
		private void updateAlpha(int iter) {
			int currentActual = actualWordCount.addAndGet(wordCount - lastWordCount);
			lastWordCount = wordCount;
			
			// Degrade the learning rate linearly towards 0 but keep a minimum
			alpha = config.initialLearningRate * Math.max(
					1 - currentActual / (double)(config.iterations * numTrainedTokens),
					0.0001
				);
			
			listener.update(
					Stage.TRAIN_NEURAL_NETWORK,
					currentActual / (double) (config.iterations * numTrainedTokens + 1)
				);
		}
		
		void handleNegativeSampling(HuffmanNode huffmanNode) {
			for (int d = 0; d <= config.negativeSamples; d++) {
				int target;
				final int label;
				if (d == 0) {
					target = huffmanNode.idx;
					label = 1;
				} else {
					nextRandom = incrementRandom(nextRandom);
					target = table[(int) (((nextRandom >> 16) % TABLE_SIZE) + TABLE_SIZE) % TABLE_SIZE];
					if (target == 0)
						target = (int)(((nextRandom % (vocabSize - 1)) + vocabSize - 1) % (vocabSize - 1)) + 1;
					if (target == huffmanNode.idx)
						continue;
					label = 0;
				}
				int l2 = target;
				double f = 0;
				for (int c = 0; c < layer1_size; c++)
					f += neu1[c] * syn1neg[l2][c];
				final double g;
				if (f > MAX_EXP)
					g = (label - 1) * alpha;
				else if (f < -MAX_EXP)
					g = (label - 0) * alpha;
				else
					g = (label - EXP_TABLE[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
				for (int c = 0; c < layer1_size; c++)
					neu1e[c] += g * syn1neg[l2][c];
				for (int c = 0; c < layer1_size; c++)
					syn1neg[l2][c] += g * neu1[c];
			}
		}
		
		/** Update the model with the given raw sentence */
		abstract void trainSentence(List<String> unfiltered);
	}
}
