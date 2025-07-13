package com.zeus.api.controller;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

//import com.theokanning.openai.OpenAiService;
//import com.theokanning.openai.completion.chat.ChatCompletionChoice;
//import com.theokanning.openai.completion.chat.ChatCompletionRequest;
//import com.theokanning.openai.completion.chat.ChatMessage;

@RestController
@RequestMapping("/llm")
public class LlmController {

//    private final OpenAiService openAiService;
//
//    public LlmController(OpenAiService openAiService) {
//        this.openAiService = openAiService;
//    }
//
//    @GetMapping("/responder")
//    public String responder(@RequestParam String pergunta) {
//        ChatCompletionRequest request = ChatCompletionRequest.builder()
//            .model("gpt-4")
//            .messages(List.of(new ChatMessage("user", pergunta)))
//            .build();
//
//        List<ChatCompletionChoice> choices = openAiService.createChatCompletion(request).getChoices();
//        return choices.get(0).getMessage().getContent();
//    }
}