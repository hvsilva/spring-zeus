package com.zeus.api.controller;

import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.json.JSONObject;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.extern.log4j.Log4j2;

@Log4j2
@RestController
@RequestMapping("/onnx-transformer")
public class TransformerOnnxController {

    private final Map<Integer, String> vocab;

    @PostMapping("/prever")
    public Map<String, Object> prever(@RequestBody Map<String, Object> payload) throws Exception {
        Object tokensObj = payload.get("tokens");
        if (tokensObj == null || !(tokensObj instanceof List)) {
            throw new IllegalArgumentException("Campo 'tokens' ausente ou inválido no corpo da requisição.");
        }

        List<Integer> tokensList = (List<Integer>) tokensObj;
        long[] tokens = tokensList.stream().mapToLong(Integer::longValue).toArray();

        long[] positionIds = new long[tokens.length];
        long[] attentionMask = new long[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            positionIds[i] = i;
            attentionMask[i] = 1;
        }

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             
        	OrtSession session = env.createSession("src/main/resources/LLM/model.onnx", new OrtSession.SessionOptions())) {

            OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), new long[]{1, tokens.length});
            OnnxTensor positionIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(positionIds), new long[]{1, tokens.length});
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), new long[]{1, tokens.length});

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("position_ids", positionIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);

            OrtSession.Result resultado = session.run(inputs);

            Object rawLogits = resultado.get(0).getValue();
            float[][][] logits3D = (float[][][]) rawLogits;
            float[][] logits = logits3D[0];

            int lastIndex = logits.length - 1;
            float[] lastLogits = logits[lastIndex];

            int nextTokenId = -1;
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < lastLogits.length; i++) {
                if (lastLogits[i] > maxLogit) {
                    maxLogit = lastLogits[i];
                    nextTokenId = i;
                }
            }

            List<Integer> novaLista = new ArrayList<>(tokensList);
            novaLista.add(nextTokenId);

            List<String> decodedTokens = novaLista.stream()
                    .map(id -> vocab.getOrDefault(id, "<UNK>"))
                    .collect(Collectors.toList());

            String tokenDecodificado = vocab.getOrDefault(nextTokenId, "<UNK>");
            String textoDecodificado = detokenizar(decodedTokens);

            Map<String, Object> resposta = new HashMap<>();
            resposta.put("tokens", tokensList);
            resposta.put("nextTokenId", nextTokenId);
            resposta.put("tokenDecodificado", tokenDecodificado);
            resposta.put("sequenciaAtualizada", novaLista);
            resposta.put("textoDecodificado", textoDecodificado);
            
            log.info("[Resposta] : {} ", resposta);
            
            return resposta;
        }
    }
    
    public TransformerOnnxController() throws Exception {
        byte[] bytes = Files.readAllBytes(Paths.get("src/main/resources/LLM/vocab.json"));
        String json = new String(bytes, StandardCharsets.UTF_8);
        JSONObject jsonObject = new JSONObject(json);
        Map<Integer, String> map = new HashMap<>();
        for (String key : jsonObject.keySet()) {
            int index = jsonObject.getInt(key);
            map.put(index, key);
        }
        this.vocab = map;
    }

    private String detokenizar(List<String> tokens) {
        StringBuilder byteString = new StringBuilder();
        for (String token : tokens) {
            if (token.startsWith("Ġ")) {
                byteString.append(" ");
                byteString.append(token.substring(1));
            } else {
                byteString.append(token);
            }
        }
        byte[] bytes = new byte[byteString.length()];
        for (int i = 0; i < byteString.length(); i++) {
            bytes[i] = (byte) byteString.charAt(i);
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }
}