package com.medallia.word2vec.util;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.logging.Log;
import org.joda.time.Duration;
import org.joda.time.Period;
import org.joda.time.format.PeriodFormat;

import java.io.ByteArrayOutputStream;
import java.io.Serializable;
import java.util.Map;

/**
 * A timer utility that can be used to keep track of the execution time of a single-threaded task
 * composed of several subtasks. Subtasks may in turn be composed of other subtasks in a recursive
 * (i.e., a tree) fashion.
 *
 * <p>
 * If a (sub) task has the same name and all of its parents up to the root share the same name, then
 * aggregate information is going to be shown for that task: total time it took, number of times it
 * was executed and average time.
 *
 * <p>
 * It is strongly recommended that this class instances be used in a try-with-resources as the actual
 * writing of the information to the logs happens on {@link #close()} and abnormal termination may
 * otherwise prevent this method from being called.
 *
 * <p>
 * This class is thread safe. However, activity tracked in threads other than the one that created
 * the {@link ProfilingTimer} will be ignored and won't be rolled up as a tree in combination with
 * the activity from other threads.
 *
 * <pre>
 *  try (ProfilingTimer timer = ProfilingTimer.start(LOG, "processing file %s", file.getName())) {
 *  	// manually starting and finishing of a task
 *  	timer.start("uncompressing file");
 *  	// ... unzipping code here ...
 *  	timer.end();
 *
 *  	// alternatively, use a try-with-resources
 *  	try (AC ac = timer.start("decrypting file")) {
 *  		// subtasks are allowed
 *  		timer.start("analyzing public/private keys");
 *  		// ... GPG stuff here ...
 *
 *  		// convenience method for sibling tasks
 *  		timer.endAndStart("actual decryption");
 *  		// ... more GPG stuff here ...
 *  		timer.end();
 *  	}
 *  }
 *
 *  // at this point all the information is written to the log, e.g., as follows
 *  // [processing file example.txt]	total time 10s
 *  // [processing file example.txt]		[uncompressing file] took 300ms
 *  // [processing file example.txt]		[decrypting file] took 9s
 *  // [processing file example.txt]			[analyzing public/private keys] took 1s
 *  // [processing file example.txt]			[actual decryption] took 8s
 * </pre>
 */
public class ProfilingTimer implements AC {

	/**
	 * Just in case we need to disable this feature due to excessive logging
	 */
	public static volatile boolean enabled = true;

	/**
	 * When this flag is enabled we only report data about the top-level activity
	 */
	public static volatile boolean topLevelInfoOnly = true;

	/**
	 * Keeps information about a task within a {@link ProfilingTimer}. Since tasks can have multiple
	 * subtasks, this represents a tree.
	 */
	public static class ProfilingTimerNode implements Serializable {
		private static final long serialVersionUID = 7464244055073290781L;

		private static final long CLOSED = -1;

		private final String taskName;
		private String logAppendMessage = "";
		private ProfilingTimerNode parent;
		private final Map<String, ProfilingTimerNode> children = Maps.newLinkedHashMap();
		private final Log log;

		private long start = System.nanoTime();
		private long totalNanos;
		private long count;

		private ProfilingTimerNode(String taskName, ProfilingTimerNode parent, Log log) {
			this.taskName = taskName;
			if (parent != null) {
				parent.addChild(this);
			}
			this.log = log;
		}

		private void addChild(ProfilingTimerNode child) {
			if (child.parent != null) {
				throw new IllegalStateException(String.format("Child [%s] already belongs to parent [%s], can't be added to new parent [%s]",
						child.taskName, child.parent.taskName, taskName));
			}

			child.parent = this;
			children.put(child.taskName, child);
		}

		private void stop() {
			if (start != CLOSED) {
				totalNanos += System.nanoTime() - start;
				count++;
				start = CLOSED;
				if (parent == null) {
					try (AC ac = NDC.push(taskName)) {
						log(0, log);
					}
				}
			}
		}

		private void appendToLog(String logAppendMessage) {
			this.logAppendMessage += logAppendMessage;
		}

		private void log(int level, Log log) {
			writeToLog(level, totalNanos, count, parent, taskName, log, logAppendMessage);

			for (ProfilingTimerNode child : children.values()) {
				child.log(level + 1, log);
			}
		}

		private void merge(ProfilingTimerNode other) {
			Preconditions.checkState(other.start == ProfilingTimerNode.CLOSED, "Can't merge non-closed node: %s", other.taskName);
			Preconditions.checkState(start == ProfilingTimerNode.CLOSED, "Can't merge into non-closed nodes: %s", taskName);

			totalNanos += other.totalNanos;
			count += other.count;
		}
	}

	/**
	 * Null object pattern {@link ProfilingTimer} instance that does nothing at all
	 */
	public static final ProfilingTimer NONE = new ProfilingTimer(null, null, null) {
		@Override public AC start(String taskName, Object... args) { return AC.NOTHING; }
		@Override public void end() { }
		@Override public void close() { }
	};

	private final Log log;
	private final ThreadLocal<ProfilingTimerNode> current = new ThreadLocal<>();
	private final ByteArrayOutputStream serializationOutput;

	/**
	 * Starts a new profiling timer with the given process name (optional arguments can be used as in {@link String#format(String, Object...)}).
	 * When this {@link AC} is closed the profiling information will be dumped on the given log.
	 *
	 * Note that this method obeys the static {@link #topLevelInfoOnly}
	 *
	 * <p>
	 * It is highly recommended to use this in a try-with-resources block so that even if there's an abrupt termination of one of the tasks,
	 * the {@link #close()} method will always be called. Otherwise the profiling information may not make it to the log.
	 *
	 * <p>
	 * Notice that this method may return {@link #NONE} if {@link #enabled} is false.
	 */
	public static ProfilingTimer create(final Log log, final String processName, final Object... args) {
		return create(log, topLevelInfoOnly, null, processName, args);
	}

	/** Same as {@link #create(Log, String, Object...)} but logs subtasks as well */
	public static ProfilingTimer createLoggingSubtasks(final Log log, final String processName, final Object... args) {
		return create(log, false, null, processName, args);
	}

	/** Same as {@link #create(Log, String, Object...)} but includes subtasks, and instead of writing to a log, it outputs the tree in serialized form */
	public static ProfilingTimer createSubtasksAndSerialization(ByteArrayOutputStream serializationOutput, final String processName, final Object... args) {
		return create(null, false, serializationOutput, processName, args);
	}

	private static ProfilingTimer create(final Log log, boolean topLevelInfoOnly, ByteArrayOutputStream serializationOutput, final String processName, final Object... args) {
		// do not use ternary as it creates an annoying resource leak warning
		if (enabled)
			if (topLevelInfoOnly)
				return new ProfilingTimer(null, null, null) {
					MutableInt level = new MutableInt(0);
					String logAppendMessage = "";
					long startNanos = System.nanoTime();
					@Override public AC start(String taskName, Object... args) {
						if (level != null)
							level.increment();
						return new AC() {
							@Override public void close() {
								level.decrement();
							}
						};
					}
					@Override public void end() {
						level.decrement();
					}
					@Override public void close() {
						if (startNanos != ProfilingTimerNode.CLOSED) {
							String taskName = String.format(processName, args);
							try (AC ac = NDC.push(taskName)) {
								writeToLog(0, System.nanoTime() - startNanos, 1, null, taskName, log, logAppendMessage);
							}
							startNanos = ProfilingTimerNode.CLOSED;
						}
					}
					@Override public void appendToLog(String logAppendMessage) {
						if (level.intValue() == 0)
							this.logAppendMessage += logAppendMessage;
					}
				};
			else
				return new ProfilingTimer(log, serializationOutput, processName, args);
		else
			return NONE;
	}

	private ProfilingTimer(Log log, ByteArrayOutputStream serializationOutput, String processName, Object... args) {
		this.log = log;
		this.serializationOutput = serializationOutput;
		start(processName, args);
	}

	/**
	 * Append the given string to the log message of the current subtask
	 */
	public void appendToLog(String logAppendMessage) {
		ProfilingTimerNode currentNode = current.get();
		if (currentNode != null) {
			currentNode.appendToLog(logAppendMessage);
		}
	}

	/**
	 * Indicates that a new task has started. Nested tasks are supported, so this method
	 * can potentially be called various times in a row without invoking {@link #end()}.
	 *
	 * <p>
	 * Optionally, this method can be used in a try-with-resources block so that there is no
	 * need to manually invoking {@link #end()} when the task at hand finishes.
	 */
	public AC start(String taskName, Object... args) {
		final ProfilingTimerNode parent = current.get();
		current.set(findOrCreateNode(String.format(taskName, args), parent));
		return new AC() {
			@Override public void close() {
				// return to the parent that we had when this AC was created
				current.set(parent);
				// close all the elements in the subtree under current
				if (parent != null)
					stopAll(parent);
			}
			private void stopAll(ProfilingTimerNode current) {
				for (ProfilingTimerNode child : current.children.values()) {
					stopAll(child);
					child.stop();
				}
			}
		};
	}

	/**
	 * Indicates that the most recently initiated task (via {@link #start(String, Object...)}) is now finished
	 */
	public void end() {
		ProfilingTimerNode currentNode = current.get();
		if (currentNode != null) {
			currentNode.stop();
			current.set(currentNode.parent);
		}
	}

	/**
	 * Convenience method for when a task starts right after the previous one finished.
	 */
	public void endAndStart(String taskName, Object... args) {
		end();
		start(taskName, args);
	}

	@Override
	public void close() {
		ProfilingTimerNode root = current.get();
		while (current.get() != null) {
			end();
		}

		if (root != null && serializationOutput != null) {
			Common.serialize(root, serializationOutput);
		}
	}

	/** Merges the specified tree as a child under the current node. */
	public void mergeTree(ProfilingTimerNode otherRoot) {
		ProfilingTimerNode currentNode = current.get();
		Preconditions.checkNotNull(currentNode);
		mergeOrAddNode(currentNode, otherRoot);
	}

	private void mergeOrAddNode(ProfilingTimerNode parent, ProfilingTimerNode child) {
		ProfilingTimerNode nodeToBeMerged = parent.children.get(child.taskName);
		if (nodeToBeMerged == null) {
			parent.addChild(child);
			return;
		}

		nodeToBeMerged.merge(child);
		for (ProfilingTimerNode grandchild : child.children.values()) {
			mergeOrAddNode(nodeToBeMerged, grandchild);
		}
	}

	private ProfilingTimerNode findOrCreateNode(String taskName, ProfilingTimerNode parent) {
		ProfilingTimerNode node = null;
		if (parent != null) {
			node = parent.children.get(taskName);
			if (node != null) {
				node.start = System.nanoTime();
			}
		}
		if (node == null) {
			node = new ProfilingTimerNode(taskName, parent, log);
		}
		return node;
	}

	/** Writes one profiling line of information to the log */
	private static void writeToLog(int level, long totalNanos, long count, ProfilingTimerNode parent, String taskName, Log log, String logAppendMessage) {
		if (log == null) {
			return;
		}

		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < level; i++) {
			sb.append('\t');
		}
		String durationText = String.format("%s%s",
				formatElapsed(totalNanos),
				count == 1 ?
						"" :
						String.format(" across %d invocations, average: %s", count, formatElapsed(totalNanos / count)));
		String text = parent == null ?
				String.format("total time %s", durationText) :
				String.format("[%s] took %s", taskName, durationText);
		sb.append(text);
		sb.append(logAppendMessage);
		log.info(sb.toString());
	}

	/** @return a human-readable formatted string for the given amount of nanos */
	private static String formatElapsed(long nanos) {
		return String.format("%s (%6.3g nanoseconds)",
				PeriodFormat.getDefault().print(Period.millis((int)(nanos / 1000))),
				(double) nanos);
	}

}
