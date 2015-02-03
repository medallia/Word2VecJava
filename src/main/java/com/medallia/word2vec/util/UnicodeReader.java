package com.medallia.word2vec.util;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PushbackInputStream;
import java.io.Reader;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;

/**
 * Generic unicode textreader, which will use BOM 
 * to identify the encoding to be used. If BOM is not found
 * then use a given default or system encoding.
 * <p>
 * BOMs for different unicodes use this standard:
 *    00 00 FE FF = UTF-32, big-endian
 *    FF FE 00 00 = UTF-32, little-endian
 *    EF BB BF = UTF-8,
 *    FE FF = UTF-16, big-endian
 *    FF FE = UTF-16, little-endian
 * <p>
 * This piece of code is found in:<p>
 * http://koti.mbnet.fi/akini/java/unicodereader/
 * <p>
 * The decoding will be handled by the decoder returned by
 * {@link Charset#newDecoder()} if strict is set to true,
 * and exceptions will be propagated from the returned
 * {@link CharsetDecoder}.
 */
public class UnicodeReader extends Reader {
	PushbackInputStream internalIn;
	InputStreamReader   internalIn2 = null;
	String              defaultEnc;
	boolean             strict;

	private static final int BOM_SIZE = 4;

	/**
	 * Constructor
	 * @param in inputstream to be read
	 * @param defaultEnc default encoding if stream does not have 
	 * 				 BOM. Give NULL to use system-level default.
	 * @param strict invalid content will give exceptions (See {@link UnicodeReader})
	 */
	public UnicodeReader(InputStream in, String defaultEnc, boolean strict) {
		internalIn = new PushbackInputStream(in, BOM_SIZE);
		this.defaultEnc = defaultEnc;
		this.strict = strict;
	}

	/**
	 * Same as {@link #UnicodeReader(InputStream, String, boolean)}, with strict = false.
	 */
	public UnicodeReader(InputStream in, String defaultEnc) {
		this(in, defaultEnc, false);
	}

	/**
	 * @return Default encoding during constructor
	 */
	public String getDefaultEncoding() {
		return defaultEnc;
	}

	/**
	 * Get stream encoding or NULL if stream is uninitialized.
	 * Call init() or read() method to initialize it.
	 */
	public String getEncoding() {
		if (internalIn2 == null) return null;
		return internalIn2.getEncoding();
	}

	/**
	 * Read-ahead four bytes and check for BOM. Extra bytes are
	 * unread back to the stream, only BOM bytes are skipped.
	 */
	protected void init() throws IOException {
		if (internalIn2 != null) return;

		String encoding;
		byte bom[] = new byte[BOM_SIZE];
		int n, unread;
		n = internalIn.read(bom, 0, bom.length);

		if ( (bom[0] == (byte)0x00) && (bom[1] == (byte)0x00) &&
				(bom[2] == (byte)0xFE) && (bom[3] == (byte)0xFF) ) {
			encoding = "UTF-32BE";
			unread = n - 4;
		} else if ( (bom[0] == (byte)0xFF) && (bom[1] == (byte)0xFE) &&
				(bom[2] == (byte)0x00) && (bom[3] == (byte)0x00) ) {
			encoding = "UTF-32LE";
			unread = n - 4;
		} else if (  (bom[0] == (byte)0xEF) && (bom[1] == (byte)0xBB) &&
				(bom[2] == (byte)0xBF) ) {
			encoding = "UTF-8";
			unread = n - 3;
		} else if ( (bom[0] == (byte)0xFE) && (bom[1] == (byte)0xFF) ) {
			encoding = "UTF-16BE";
			unread = n - 2;
		} else if ( (bom[0] == (byte)0xFF) && (bom[1] == (byte)0xFE) ) {
			encoding = "UTF-16LE";
			unread = n - 2;
		} else {
			// Unicode BOM not found, unread all bytes
			encoding = defaultEnc;
			unread = n;
		}

		if (unread > 0) internalIn.unread(bom, (n - unread), unread);

		// Use given encoding
		if (encoding == null) {
			internalIn2 = new InputStreamReader(internalIn);
		} else if (strict) {
			internalIn2 = new InputStreamReader(internalIn, Charset.forName(encoding).newDecoder());
		} else {
			internalIn2 = new InputStreamReader(internalIn, encoding);
		}
	}

	@Override public void close() throws IOException {
		init();
		internalIn2.close();
	}

	@Override public int read(char[] cbuf, int off, int len) throws IOException {
		init();
		return internalIn2.read(cbuf, off, len);
	}

	@Override public boolean ready() throws IOException {
		init();
		return internalIn2.ready();
	}
}