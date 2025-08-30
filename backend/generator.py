"""
Answer generation module.
Handles response generation using OpenAI API and OpenRouter with retrieved context.
"""

import openai
import requests
import json
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class AnswerGenerator:
    """Generates answers using OpenAI models and OpenRouter with retrieved context."""
    
    def __init__(self, openai_api_key: str = None, openrouter_api_key: str = None, 
                 model_name: str = "gpt-3.5-turbo", temperature: float = 0.1, 
                 provider: str = "openai"):
        """
        Initialize the answer generator.
        
        Args:
            openai_api_key: OpenAI API key
            openrouter_api_key: OpenRouter API key
            model_name: Model to use for generation
            temperature: Temperature for response generation
            provider: Provider to use ('openai' or 'openrouter')
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.conversation_history = []
        
        if provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI provider")
            self.client = ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name=model_name,
                temperature=temperature
            )
        elif provider == "openrouter":
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key required for OpenRouter provider")
            self.openrouter_api_key = openrouter_api_key
            self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            raise ValueError("Provider must be 'openai' or 'openrouter'")
    
    def _call_openrouter(self, messages: List[Dict[str, str]]) -> str:
        """
        Make API call to OpenRouter.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generated response text
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature
        }
        
        response = requests.post(self.openrouter_url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

    def generate_answer(self, query: str, context: str, include_sources: bool = True) -> str:
        """
        Generate answer based on query and retrieved context.
        
        Args:
            query: User question
            context: Retrieved context from documents
            include_sources: Whether to include source information
            
        Returns:
            Generated answer string
        """
        try:
            # Create system prompt
            system_prompt = self._create_system_prompt(include_sources)
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(query, context)
            
            if self.provider == "openai":
                # Generate response using OpenAI
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = self.client(messages)
                answer = response.content
                
            elif self.provider == "openrouter":
                # Generate response using OpenRouter
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                answer = self._call_openrouter(messages)
            
            # Store in conversation history
            self.conversation_history.append({
                'query': query,
                'context': context,
                'answer': answer
            })
            
            return answer
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_conversational_answer(self, query: str, context: str, 
                                     conversation_history: List[Dict] = None) -> str:
        """
        Generate conversational answer considering chat history.
        
        Args:
            query: Current user question
            context: Retrieved context from documents
            conversation_history: Previous conversation turns
            
        Returns:
            Generated answer string
        """
        try:
            # Create system prompt
            system_prompt = self._create_conversational_system_prompt()
            
            if self.provider == "openai":
                # Build conversation messages for OpenAI
                messages = [SystemMessage(content=system_prompt)]
                
                # Add conversation history
                if conversation_history:
                    for turn in conversation_history[-5:]:  # Keep last 5 turns
                        messages.append(HumanMessage(content=turn.get('query', '')))
                        messages.append(AIMessage(content=turn.get('answer', '')))
                
                # Add current query with context
                current_prompt = self._create_user_prompt(query, context)
                messages.append(HumanMessage(content=current_prompt))
                
                # Generate response
                response = self.client(messages)
                return response.content
                
            elif self.provider == "openrouter":
                # Build conversation messages for OpenRouter
                messages = [{"role": "system", "content": system_prompt}]
                
                # Add conversation history
                if conversation_history:
                    for turn in conversation_history[-5:]:  # Keep last 5 turns
                        user_query = turn.get('query', '')
                        assistant_response = turn.get('answer', '')
                        if user_query:
                            messages.append({"role": "user", "content": user_query})
                        if assistant_response:
                            messages.append({"role": "assistant", "content": assistant_response})
                
                # Add current query with context
                current_prompt = self._create_user_prompt(query, context)
                messages.append({"role": "user", "content": current_prompt})
                
                # Generate response
                return self._call_openrouter(messages)
        
        except Exception as e:
            return f"Error generating conversational answer: {str(e)}"
    
    def _create_system_prompt(self, include_sources: bool = True) -> str:
        """Create system prompt for answer generation."""
        base_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        
Your responsibilities:
1. Answer questions accurately using only the information from the provided context
2. If the context doesn't contain enough information to answer the question, clearly state this
3. Be concise but comprehensive in your responses
4. Maintain a professional and helpful tone
5. Do not make up information that is not in the context"""

        if include_sources:
            base_prompt += "\n6. When possible, indicate which part of the context supports your answer"
        
        return base_prompt
    
    def _create_conversational_system_prompt(self) -> str:
        """Create system prompt for conversational interaction."""
        return """You are a helpful AI assistant engaged in a conversation about documents. 
        
Your responsibilities:
1. Answer questions based on the provided context and conversation history
2. Maintain context awareness across the conversation
3. Reference previous parts of the conversation when relevant
4. If information is not available in the context, clearly state this
5. Be conversational but accurate
6. Build upon previous answers when appropriate"""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create user prompt with query and context."""
        return f"""Based on the following context, please answer the question:

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
    
    def summarize_document(self, content: str, max_length: int = 500) -> str:
        """
        Generate a summary of document content.
        
        Args:
            content: Document content to summarize
            max_length: Maximum length of summary
            
        Returns:
            Generated summary string
        """
        try:
            prompt = f"""Please provide a concise summary of the following document content in no more than {max_length} characters:

{content}

Summary:"""
            
            messages = [HumanMessage(content=prompt)]
            response = self.client(messages)
            return response.content
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def suggest_follow_up_questions(self, query: str, answer: str, context: str) -> List[str]:
        """
        Suggest follow-up questions based on the current query and answer.
        
        Args:
            query: Original user query
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            List of suggested follow-up questions
        """
        try:
            prompt = f"""Based on the following question, answer, and context, suggest 3 relevant follow-up questions:

Original Question: {query}
Answer: {answer}
Context: {context[:1000]}...

Please provide 3 specific follow-up questions that would help the user explore this topic further:"""

            if self.provider == "openai":
                messages = [HumanMessage(content=prompt)]
                response = self.client(messages)
                response_text = response.content
            elif self.provider == "openrouter":
                messages = [{"role": "user", "content": prompt}]
                response_text = self._call_openrouter(messages)
            
            # Parse suggestions (assuming they're returned as numbered list)
            suggestions = []
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering and clean up
                    clean_question = line.lstrip('123456789.- •').strip()
                    if clean_question:
                        suggestions.append(clean_question)
            
            return suggestions[:3]  # Return max 3 suggestions
        
        except Exception as e:
            return [f"Error generating suggestions: {str(e)}"]


class StreamingAnswerGenerator(AnswerGenerator):
    """Extended generator with streaming capabilities for real-time responses."""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        super().__init__(openai_api_key, model_name, temperature)
        # Initialize streaming client
        self.streaming_client = openai.OpenAI(api_key=openai_api_key)
    
    def generate_streaming_answer(self, query: str, context: str):
        """
        Generate streaming answer for real-time display.
        
        Args:
            query: User question
            context: Retrieved context
            
        Yields:
            Chunks of the generated response
        """
        try:
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(query, context)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            stream = self.streaming_client.chat.completions.create(
                model=self.client.model_name,
                messages=messages,
                temperature=self.client.temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            yield f"Error generating streaming answer: {str(e)}"