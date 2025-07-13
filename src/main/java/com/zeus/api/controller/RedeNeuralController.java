package com.zeus.api.controller;

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
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/rede-neural")
public class RedeNeuralController {

	private MultiLayerNetwork model;

	@GetMapping("/treinar")
	public String treinar() {
		double[][] input = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		double[][] output = new double[][] { { 0 }, { 1 }, { 1 }, { 0 } }; // XOR

		DataSet dataset = new DataSet(Nd4j.create(input), Nd4j.create(output));

		MultiLayerConfiguration config = new NeuralNetConfiguration.Builder().seed(123).updater(new Nesterovs(0.1, 0.9))
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
				.layer(new DenseLayer.Builder().nIn(2).nOut(4).activation(Activation.RELU).build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nOut(1)
						.build())
				.build();

		model = new MultiLayerNetwork(config);
		model.init();
		model.setListeners(new ScoreIterationListener(10));

		for (int i = 0; i < 1000; i++) {
			model.fit(new ListDataSetIterator<>(dataset.asList(), 4));
		}

		return "Rede neural treinada para XOR";
	}

	@GetMapping("/predict")
	public Map<String, Object> prever(@RequestParam double x1, @RequestParam double x2) {
		Map<String, Object> resultado = new HashMap<>();

		if (model == null) {
			resultado.put("erro", "A rede ainda não foi treinada. Chame /treinar primeiro.");
			return resultado;
		}

		INDArray input = Nd4j.create(new double[] { x1, x2 }, new int[] { 1, 2 });
		INDArray output = model.output(input);

		resultado.put("entrada", new double[] { x1, x2 });
		resultado.put("saida_prevista", output.getDouble(0));
		return resultado;
	}

}