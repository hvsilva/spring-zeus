package com.zeus.api.controller;

import org.springframework.web.bind.annotation.*;

import com.zeus.api.test.PerceptronSimples;

import java.util.*;

@RestController
@RequestMapping("/perceptron")
public class PerceptronController {

    private PerceptronSimples perceptron;

    @PostMapping("/treinar")
    public String treinar(@RequestBody Map<String, Object> payload) {
        List<List<Double>> entradasList = (List<List<Double>>) payload.get("entradas");
        List<Integer> saidasList = (List<Integer>) payload.get("saidas");
        int epocas = (int) payload.getOrDefault("epocas", 10);

        double[][] entradas = entradasList.stream()
                .map(l -> l.stream().mapToDouble(Double::doubleValue).toArray())
                .toArray(double[][]::new);

        int[] saidas = saidasList.stream().mapToInt(Integer::intValue).toArray();

        perceptron = new PerceptronSimples(entradas[0].length);
        perceptron.treinar(entradas, saidas, epocas);

        return "Perceptron treinado com sucesso.";
    }

    @PostMapping("/prever")
    public Map<String, Object> prever(@RequestBody Map<String, List<Double>> payload) {
        List<Double> entradaList = payload.get("entrada");
        double[] entrada = entradaList.stream().mapToDouble(Double::doubleValue).toArray();

        int resultado = perceptron.prever(entrada);

        Map<String, Object> resposta = new HashMap<>();
        resposta.put("entrada", entrada);
        resposta.put("classificacao", resultado);
        resposta.put("pesos", perceptron.getPesos());
        resposta.put("bias", perceptron.getBias());

        return resposta;
    }
}