package main.java.com.devon;

import opennlp.tools.doccat.*;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.*;
import opennlp.tools.util.model.ModelUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Collectors;

public class OpenNLPAIChatBot {
    private static Logger logger = LogManager.getLogger(OpenNLPAIChatBot.class);
    private static Map<String, String> questionAnswer = new HashMap<>();
    private static Map<String, String> modelFileName = new HashMap<>();

    // Define answers for each given category.
    static {
        questionAnswer.put("greeting", "Hello, my name is Stacy.  How may I help you today?");
        questionAnswer.put("product-inquiry",
                "Our company sells Auto, Life, and Homeowners Insurance to help protect you, and your loved ones.");
        questionAnswer.put("price-inquiry", "The price is competitive, based on your specific needs, and coverage options.");
        questionAnswer.put("contact-inquiry", "Please free to reach us via telephone at 1-800-555-sold.");
        questionAnswer.put("conversation-continue", "What else can I help you with?");
        questionAnswer.put("conversation-complete", "It was nice chatting with you. Goodbye!");
    }

    static {
        modelFileName.put("sentenceDetectorME", "en-sent.bin");
        modelFileName.put("tokenizerME", "en-token.bin");
        modelFileName.put("posTaggerME", "en-pos-maxent.bin");
        modelFileName.put("lemmatizerME", "en-lemmatizer.bin");
    }
    static SentenceDetectorME sentenceDetectorME;
    static TokenizerME tokenizerME;
    static POSTaggerME posTaggerME;
    static LemmatizerME lemmatizerME;

    public static void doInitialize() {

        try {
            logger.info("Current log level is set to : " + logger.getLevel());
            // Used for splitting up raw text into sentences.
            try (InputStream modelIn = new FileInputStream(new File(
                    OpenNLPAIChatBot.class.getClassLoader().getResource(modelFileName.get("sentenceDetectorME")).getFile()))) {
                sentenceDetectorME = new SentenceDetectorME(new SentenceModel(modelIn));
            } catch (Exception ex) {
                logger.error("Exception while attempting to load " + modelFileName.get("sentenceDetectorME"));
                ex.printStackTrace();
            }

            // Used for converting raw text into separated tokens.
            try (InputStream modelIn = new FileInputStream(new File(
                    OpenNLPAIChatBot.class.getClassLoader().getResource(modelFileName.get("tokenizerME")).getFile()))) {
                tokenizerME = new TokenizerME(new TokenizerModel(modelIn));
            } catch (Exception ex) {
                logger.error("Exception while attempting to load " + modelFileName.get("tokenizerME"));
                ex.printStackTrace();
            }

            // Tries to predict whether words are nouns, verbs, or any of 70 other POS tags
            // depending on their surrounding context.
            try (InputStream modelIn = new FileInputStream(new File(
                    OpenNLPAIChatBot.class.getClassLoader().getResource(modelFileName.get("posTaggerME")).getFile()))) {
                posTaggerME = new POSTaggerME(new POSModel(modelIn));
            } catch (Exception ex) {
                logger.error("Exception while attempting to load " + modelFileName.get("posTaggerME"));
                ex.printStackTrace();
            }

            // Tries to predict the induced permutation class for each word depending on
            // its surrounding context.
            try (InputStream modelIn = new FileInputStream(new File(
                    OpenNLPAIChatBot.class.getClassLoader().getResource(modelFileName.get("lemmatizerME")).getFile()))) {
                lemmatizerME = new LemmatizerME(new LemmatizerModel(modelIn));
            } catch (Exception ex) {
                logger.error("Exception while attempting to load " + modelFileName.get("lemmatizerME"));
                ex.printStackTrace();
            }
        } catch (Exception ex) {
            logger.error("Exception while initializing... ");
            ex.printStackTrace();
        } finally {
            if (sentenceDetectorME == null || tokenizerME == null || posTaggerME == null || lemmatizerME == null) {
                logger.error("Not all models were loaded.  Check model filename/location.");
                System.exit(-1);
            } else
                logger.info("All models loaded successfully!");
        }

    }

    public static void main(String[] args) throws FileNotFoundException, IOException, InterruptedException {

        // Train categorizer model to the training data we created.
        DoccatModel model = trainCategorizerModel();
        doInitialize();

        // Take chat inputs from console (user) in a loop.
        Scanner scanner = new Scanner(System.in);
        while (true) {

            // Get chat input from user.
            System.out.print("##### You:  ");
            String userInput = scanner.nextLine();

            // Break users chat input into sentences using sentence detection.
            String[] sentences = breakSentences(userInput);

            String answer = "";
            boolean conversationComplete = false;

            // Loop through sentences.
            for (String sentence : sentences) {

                // Separate words from each sentence using tokenizer.
                String[] tokens = tokenizeSentence(sentence);

                // Tag separated words with POS tags to understand their gramatical structure.
                String[] posTags = detectPOSTags(tokens);

                // Lemmatize each word so that its easy to categorize.
                String[] lemmas = lemmatizeTokens(tokens, posTags);

                // Determine BEST category using lemmatized tokens used a mode that we trained
                // at start.
                String category = detectCategory(model, lemmas);

                // Get predefined answer from given category & add to answer.
                answer = answer + " " + questionAnswer.get(category);

                // If category conversation-complete, we will end chat conversation.
                if ("conversation-complete".equals(category)) {
                    conversationComplete = true;
                }
            }

            // Display the answer back to the user. If conversation is marked as complete, then end
            // loop & program.
            System.out.println("##### Virtual Agent: " + answer);
            if (conversationComplete) {
                break;
            }

        }

    }

    /**
     * Train categorizer model as per the category sample training data we created.
     *
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static DoccatModel trainCategorizerModel() {
        // documentcategorizer.txt is a custom training data file with categories as necessary for our chat
        // requirements.
        DoccatModel model = null;
        String modelFileNameString = "documentcategorizer.txt";
        try  {
            File file = new File(
                    OpenNLPAIChatBot.class.getClassLoader().getResource(modelFileNameString).getFile());
            InputStreamFactory inputStreamFactory = new MarkableFileInputStreamFactory(file);
            ObjectStream<String> lineStream = new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
            ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

            DoccatFactory factory = new DoccatFactory(new FeatureGenerator[] { new BagOfWordsFeatureGenerator() });

            TrainingParameters params = ModelUtil.createDefaultTrainingParameters();
            params.put(TrainingParameters.CUTOFF_PARAM, 0);

            // Train a model with classifications from above file.
            model = DocumentCategorizerME.train("en", sampleStream, params, factory);
            //return model;
        } catch (Exception ex) {
            logger.error("Exception while attempting to train Categorizer Model using: " + modelFileNameString);
            ex.printStackTrace();
        } finally {
            if (model == null) {
                logger.error("Unable to train Categorizer model.  Check model filename/location.");
                System.exit(-1);
            } else
                logger.info("Categorizer model trained successfully!");
        }
        return model;
    }

    /**
     * Detect category using given token. Use categorizer feature of Apache OpenNLP.
     *
     * @param model
     * @param finalTokens
     * @return
     * @throws IOException
     */
    private static String detectCategory(DoccatModel model, String[] finalTokens) throws IOException {

        // Initialize document categorizer tool
        DocumentCategorizerME myCategorizer = new DocumentCategorizerME(model);

        // Get best possible category.
        double[] probabilitiesOfOutcomes = myCategorizer.categorize(finalTokens);
        String category = myCategorizer.getBestCategory(probabilitiesOfOutcomes);
        logger.debug("Category: " + category);

        return category;
    }

    /**
     * Break data into sentences using sentence detection feature of Apache OpenNLP.
     *
     * @param data
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static String[] breakSentences(String data) {
        String[] sentences = sentenceDetectorME.sentDetect(data);
        logger.debug("Sentence Detection: " + Arrays.stream(sentences).collect(Collectors.joining(" | ")));

        return sentences;
    }

    /**
     * Break sentence into words & punctuation marks using tokenizer feature of
     * Apache OpenNLP.
     *
     * @param sentence
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static String[] tokenizeSentence(String sentence) {
        // Tokenize sentence.
        String[] tokens = tokenizerME.tokenize(sentence);
        logger.debug("Tokenizer : " + Arrays.stream(tokens).collect(Collectors.joining(" | ")));

        return tokens;
    }

    /**
     * Find part-of-speech or POS tags of all tokens using POS tagger feature of
     * Apache OpenNLP.
     *
     * @param tokens
     * @return
     * @throws IOException
     */
    private static String[] detectPOSTags(String[] tokens) throws IOException {
        // Tag sentence.
        String[] posTokens = posTaggerME.tag(tokens);
        logger.debug("POS Tags : " + Arrays.stream(posTokens).collect(Collectors.joining(" | ")));

        return posTokens;
    }

    /**
     * Find lemma of tokens using lemmatizer feature of Apache OpenNLP.
     *
     * @param tokens
     * @param posTags
     * @return
     * @throws InvalidFormatException
     * @throws IOException
     */
    private static String[] lemmatizeTokens(String[] tokens, String[] posTags)
            throws InvalidFormatException, IOException {
        String[] lemmaTokens = lemmatizerME.lemmatize(tokens, posTags);
        logger.debug("Lemmatizer : " + Arrays.stream(lemmaTokens).collect(Collectors.joining(" | ")));

        return lemmaTokens;
    }
}
