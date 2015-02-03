package com.medallia.word2vec.util;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import com.google.common.base.Function;

import com.google.common.base.Strings;

/**
 * Collection of file-related utilities.
 */
public final class FileUtils {

	public static final int ONE_KB = 1<<10;
	public static final int ONE_MB = 1<<20;

	public static final Function<File, String> FILE_TO_NAME = new Function<File, String>() {
		@Override public String apply(File file) {
			return file.getName();
		}
	};

	/**
	 * Returns a subdirectory of a given directory; the subdirectory is expected to already exist.
	 * @param parent the directory in which to find the specified subdirectory
	 * @param item the name of the subdirectory
	 * @return the subdirectory having the specified name; null if no such directory exists or
	 * exists but is a regular file.
	 */
	public static File getDir(File parent, String item) {
		File dir = new File(parent, item);
		return (dir.exists() && dir.isDirectory()) ? dir : null;
	}

	/** @return File for the specified directory; creates the directory if necessary. */
	private static File getOrCreateDir(File f) throws IOException {
		if (!f.exists()) {
			if (!f.mkdirs()) {
				throw new IOException(f.getName() + ": Unable to create directory: " + f.getAbsolutePath());
			}
		} else if (!f.isDirectory()) {
			throw new IOException(f.getName() + ": Exists and is not a directory: " + f.getAbsolutePath());
		}
		return f;
	}

	/**
	 * @return File for the specified directory, creates dirName directory if necessary.
	 * @throws IOException if there is an error during directory creation or if a non-directory file with the desired name already
	 *                     exists.
	 */
	public static File getOrCreateDir(String dirName) throws IOException {
		return getOrCreateDir(new File(dirName));
	}

	/**
	 * @return File for the specified directory; parent must already exist, creates dirName subdirectory if necessary.
	 * @throws IOException if there is an error during directory creation or if a non-directory file with the desired name already
	 *                     exists.
	 */
	public static File getOrCreateDir(File parent, String dirName) throws IOException {
		return getOrCreateDir(new File(parent, dirName));
	}

	/**
	 * @return File for the specified directory; parent must already exist, creates all intermediate dirNames subdirectories if necessary.
	 * @throws IOException if there is an error during directory creation or if a non-directory file with the desired name already
	 *                     exists.
	 */
	public static File getOrCreateDir(File parent, String ... dirNames) throws IOException {
		return getOrCreateDir(Paths.get(parent.getPath(), dirNames).toFile());
	}

	/**
	 * Deletes a file or directory.
	 * If the file is a directory it recursively deletes it. 
	 * @param file file to be deleted
	 * @return true if all the files where deleted successfully.
	 */
	public static boolean deleteRecursive(final File file) {
		boolean result = true;
		if (file.isDirectory()) {
			for (final File inner : file.listFiles()) {
				result &= deleteRecursive(inner);
			}
		}
		return result & file.delete();
	}

	/** Utility class; don't instantiate. */
	private FileUtils() {
		throw new AssertionError("Do not instantiate.");
	}

	/** @return A random temporary folder that can be used for file-system operations testing */
	public static File getRandomTemporaryFolder(String prefix, String suffix) {
		return new File(System.getProperty("java.io.tmpdir"), Strings.nullToEmpty(prefix) + UUID.randomUUID().toString() + Strings.nullToEmpty(suffix));
	}
}
