package com.medallia.word2vec;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.medallia.word2vec.Searcher.Match;
import com.medallia.word2vec.Searcher.UnknownWordException;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import com.medallia.word2vec.thrift.Word2VecModelThrift;
import com.medallia.word2vec.util.Common;
import com.medallia.word2vec.util.ThriftUtils;
import org.apache.commons.io.FileUtils;
import org.apache.thrift.TException;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** 
 * Tests for {@link Word2VecModel} and related classes.
 * <p>
 * Note that the implementation is expected to be deterministic if numThreads is set to 1
 */
public class Word2VecTest {
	@Rule
	public ExpectedException expected = ExpectedException.none();
	
	/** Clean up after a test run */
	@After
	public void after() {
		// Unset the interrupted flag to avoid polluting other tests
		Thread.interrupted();
	}
	
	/** Test {@link NeuralNetworkType#CBOW} */
	@Test
	public void testCBOW() throws IOException, TException, InterruptedException {
		assertModelMatches("cbowBasic.model",
				Word2VecModel.trainer()
						.setMinVocabFrequency(6)
						.useNumThreads(1)
						.setWindowSize(8)
						.type(NeuralNetworkType.CBOW)
						.useHierarchicalSoftmax()
						.setLayerSize(25)
						.setDownSamplingRate(1e-3)
						.setNumIterations(1)
						.train(testData())
		);
	}
	
	/** Test {@link NeuralNetworkType#CBOW} with 15 iterations */
	@Test
	public void testCBOWwith15Iterations() throws IOException, TException, InterruptedException {
		assertModelMatches("cbowIterations.model",
				Word2VecModel.trainer()
					.setMinVocabFrequency(5)
					.useNumThreads(1)
					.setWindowSize(8)
					.type(NeuralNetworkType.CBOW)
					.useHierarchicalSoftmax()
					.setLayerSize(25)
					.useNegativeSamples(5)
					.setDownSamplingRate(1e-3)
					.setNumIterations(15)
					.train(testData())
			);
	}
	
	/** Test {@link NeuralNetworkType#SKIP_GRAM} */
	@Test
	public void testSkipGram() throws IOException, TException, InterruptedException {
		assertModelMatches("skipGramBasic.model",
				Word2VecModel.trainer()
					.setMinVocabFrequency(6)
					.useNumThreads(1)
					.setWindowSize(8)
					.type(NeuralNetworkType.SKIP_GRAM)
					.useHierarchicalSoftmax()
					.setLayerSize(25)
					.setDownSamplingRate(1e-3)
					.setNumIterations(1)
					.train(testData())
			);
	}
	
	/** Test {@link NeuralNetworkType#SKIP_GRAM} with 15 iterations */
	@Test
	public void testSkipGramWith15Iterations() throws IOException, TException, InterruptedException {
		assertModelMatches("skipGramIterations.model",
				Word2VecModel.trainer()
					.setMinVocabFrequency(6)
					.useNumThreads(1)
					.setWindowSize(8)
					.type(NeuralNetworkType.SKIP_GRAM)
					.useHierarchicalSoftmax()
					.setLayerSize(25)
					.setDownSamplingRate(1e-3)
					.setNumIterations(15)
					.train(testData())
			);
	}
	
	/** Test that we can interrupt the huffman encoding process */
	@Test
	public void testInterruptHuffman() throws IOException, InterruptedException {
		expected.expect(InterruptedException.class);
		trainer()
			.type(NeuralNetworkType.SKIP_GRAM)
			.setNumIterations(15)
			.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						if (stage == Stage.CREATE_HUFFMAN_ENCODING)
							Thread.currentThread().interrupt();
						else if (stage == Stage.TRAIN_NEURAL_NETWORK)
							fail("Should not have reached this stage");
					}
				})
			.train(testData());
	}

	/** Test that we can interrupt the neural network training process */
	@Test
	public void testInterruptNeuralNetworkTraining() throws InterruptedException, IOException {
		expected.expect(InterruptedException.class);
		trainer()
			.type(NeuralNetworkType.SKIP_GRAM)
			.setNumIterations(15)
			.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						if (stage == Stage.TRAIN_NEURAL_NETWORK)
							Thread.currentThread().interrupt();
					}
				})
			.train(testData());
	}
	
	/** 
	 * Test the search results are deterministic
	 * Note the actual values may not make sense since the model we train isn't tuned
	 */
	@Test
	public void testSearch() throws InterruptedException, IOException, UnknownWordException {
		Word2VecModel model = trainer()
			.type(NeuralNetworkType.SKIP_GRAM)
			.train(testData());
		
		List<Match> matches = model.forSearch().getMatches("anarchism", 5);
		
		assertEquals(
				ImmutableList.of("anarchism", "feminism", "trouble", "left", "capitalism"),
				Lists.transform(matches, Match.TO_WORD)
			);
	}
	
	/** @return {@link Word2VecTrainer} which by default uses all of the supported features */
	@VisibleForTesting
	public static Word2VecTrainerBuilder trainer() {
		return Word2VecModel.trainer()
			.setMinVocabFrequency(6)
			.useNumThreads(1)
			.setWindowSize(8)
			.type(NeuralNetworkType.CBOW)
			.useHierarchicalSoftmax()
			.setLayerSize(25)
			.setDownSamplingRate(1e-3)
			.setNumIterations(1);
	}

	/** @return raw test dataset. The tokens are separated by newlines. */
	@VisibleForTesting
	public static Iterable<List<String>> testData() throws IOException {
		List<String> lines = Common.readResource(Word2VecTest.class, "word2vec.short.txt");
		Iterable<List<String>> partitioned = Iterables.partition(lines, 1000);
		return partitioned;
	}
	
	private void assertModelMatches(String expectedResource, Word2VecModel model) throws TException {
		final String thrift;
		try {
			thrift = Common.readResourceToStringChecked(getClass(), expectedResource);
		} catch (IOException ioe) {
			String filename = "/tmp/" + expectedResource;
			try {
				FileUtils.writeStringToFile(
						new File(filename),
						ThriftUtils.serializeJson(model.toThrift())
				);
			} catch (IOException e) {
				throw new AssertionError("Could not read resource " + expectedResource + " and could not write expected output to /tmp");
			}
			throw new AssertionError("Could not read resource " + expectedResource + " wrote to " + filename);
		}
		
		Word2VecModelThrift expected = ThriftUtils.deserializeJson(
				new Word2VecModelThrift(),
				thrift
		);
		
		assertEquals("Mismatched vocab", expected.getVocab().size(), Iterables.size(model.getVocab()));
		
		assertEquals(expected, model.toThrift());
	}
}
