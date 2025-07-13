package com.zeus.api.controller;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/rede-neural")
public class RedeNeuralController {

	private MultiLayerNetwork model;
	private static final String MODELO_PATH = "modelo-xor.zip";
	private static final String MODELO_NUMERICO_PATH = "modelo-numerico.zip";

	@GetMapping("/treinar")
	public String treinar() {
		double[][] input = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		double[][] output = new double[][] { { 0 }, { 1 }, { 1 }, { 0 } }; // XOR

		DataSet dataset = new DataSet(Nd4j.create(input), Nd4j.create(output));

		MultiLayerConfiguration config = new NeuralNetConfiguration.Builder().seed(123)
				.updater(new Nesterovs(0.1, 0.9))
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
				.layer(new DenseLayer.Builder().nIn(2).nOut(4).activation(Activation.RELU).build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nOut(1).build())						
				.build();

		model = new MultiLayerNetwork(config);
		model.init();
		model.setListeners(new ScoreIterationListener(10));

		for (int i = 0; i < 1000; i++) {
			model.fit(new ListDataSetIterator<>(dataset.asList(), 4));
		}

		try {
			model.save(new File(MODELO_PATH), true);
		} catch (Exception e) {
			return "Erro ao salvar o modelo: " + e.getMessage();
		}

		return "Rede neural treinada e salva em disco.";
	}

	@PostMapping("/predict")
	public Map<String, Object> prever(@RequestBody Map<String, Double> entradaJson) {
		Map<String, Object> resultado = new HashMap<>();

		try {
			if (model == null) {
				File f = new File(MODELO_PATH);
				if (f.exists()) {
					model = MultiLayerNetwork.load(f, true);
				} else {
					resultado.put("erro", "A rede ainda não foi treinada. Chame treinar primeiro.");
					return resultado;
				}
			}

			double x1 = entradaJson.getOrDefault("x1", 0.0);
			double x2 = entradaJson.getOrDefault("x2", 0.0);
			INDArray input = Nd4j.create(new double[] { x1, x2 }, new int[] { 1, 2 });
			INDArray output = model.output(input);
			double saida = output.getDouble(0);

			resultado.put("entrada", new double[] { x1, x2 });
			resultado.put("saida_prevista", saida);
			resultado.put("classificacao", saida >= 0.5 ? 1 : 0);
		} catch (Exception e) {
			resultado.put("erro", "Falha ao prever: " + e.getMessage());
		}

		return resultado;
	}

	@PostMapping("/treinar-numerico")
	public String treinarNumerico(@RequestBody Map<String, double[][]> payload) {
		double[][] input = payload.get("input");
		double[][] output = payload.get("output");

		if (input == null || output == null || input.length != output.length) {
			return "Dados de entrada/saída inválidos.";
		}

		DataSet dataset = new DataSet(Nd4j.create(input), Nd4j.create(output));

		int entradas = input[0].length;
		int saidas = output[0].length;

		MultiLayerConfiguration config = new NeuralNetConfiguration.Builder().seed(1234)
				.updater(new Nesterovs(0.01, 0.9))
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
				.layer(new DenseLayer.Builder().nIn(entradas).nOut(10).activation(Activation.RELU).build())				
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nOut(saidas).build())						
				.build();

		model = new MultiLayerNetwork(config);
		model.init();
		model.setListeners(new ScoreIterationListener(10));

		for (int i = 0; i < 1000; i++) {
			model.fit(new ListDataSetIterator<>(dataset.asList(), 10));
		}

		try {
			model.save(new File(MODELO_NUMERICO_PATH), true);
		} catch (Exception e) {
			return "Erro ao salvar modelo numérico: " + e.getMessage();
		}

		return "Modelo numérico treinado e salvo com sucesso.";
	}

	@PostMapping("/predict-numerico")
	public Map<String, Object> preverNumerico(@RequestBody Map<String, double[]> entradaJson) {
		Map<String, Object> resultado = new HashMap<>();

		try {
			if (model == null) {
				File f = new File(MODELO_NUMERICO_PATH);
				if (f.exists()) {
					model = MultiLayerNetwork.load(f, true);
				} else {
					resultado.put("erro", "O modelo numérico ainda não foi treinado. Chame treinar-numerico primeiro.");

					return resultado;
				}
			}

			double[] valores = entradaJson.get("entrada");
			INDArray input = Nd4j.create(valores, new int[] { 1, valores.length });
			INDArray output = model.output(input);

			resultado.put("entrada", valores);
			resultado.put("saida_prevista", output.toDoubleVector());
		} catch (Exception e) {
			resultado.put("erro", "Falha ao prever (modelo numérico): " + e.getMessage());
		}

		return resultado;
	}
}