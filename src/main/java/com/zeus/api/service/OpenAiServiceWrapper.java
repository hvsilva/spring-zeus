package com.zeus.api.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

//import com.theokanning.openai.OpenAiService;
//import com.theokanning.openai.completion.chat.ChatCompletionChoice;
//import com.theokanning.openai.completion.chat.ChatCompletionRequest;
//import com.theokanning.openai.completion.chat.ChatMessage;

//@Service
public class OpenAiServiceWrapper {

//    private final OpenAiService openAiService;
//
//    public OpenAiServiceWrapper(@Value("${openai.api.key}") String apiKey) {
//        this.openAiService = new OpenAiService(apiKey);
//    }
//
//    public String perguntar(String prompt) {
//        ChatCompletionRequest request = ChatCompletionRequest.builder()
//                .model("gpt-4")
//                .messages(List.of(new ChatMessage("user", prompt)))
//                .build();
//
//        List<ChatCompletionChoice> choices = openAiService.createChatCompletion(request).getChoices();
//        return choices.get(0).getMessage().getContent();
//    }
}