document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatContainer = document.getElementById('chat-container');
    const ingestBtn = document.getElementById('ingest-btn');
    
    // Auto focus input
    userInput.focus();
    
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to UI
        appendMessage(message, 'user');
        userInput.value = '';
        
        // Add loading indicator
        const loadingId = appendLoadingIndicator();
        
        try {
            // Send to backend
            const response = await fetch('http://127.0.0.1:8000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });
            
            // Remove loading indicator
            removeElement(loadingId);
            
            if (response.ok) {
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.includes("application/json")) {
                    const data = await response.json();
                    appendMessage(data.reply || data.error, 'system');
                } else {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder('utf-8');
                    
                    const msgDiv = document.createElement('div');
                    msgDiv.className = `message system-message`;
                    const avatarDiv = document.createElement('div');
                    avatarDiv.className = `avatar system-avatar`;
                    avatarDiv.innerHTML = '<i class="fa-solid fa-robot"></i>';
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'message-content';
                    contentDiv.innerHTML = `<p></p>`;
                    msgDiv.appendChild(avatarDiv);
                    msgDiv.appendChild(contentDiv);
                    chatContainer.appendChild(msgDiv);
                    
                    let fullText = "";
                    const pElement = contentDiv.querySelector('p');
                    
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value, { stream: true });
                        fullText += chunk;
                        
                        // Simple markdown parsing for bold and line breaks
                        const formattedText = fullText
                            .replace(/\n/g, '<br>')
                            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                            
                        pElement.innerHTML = formattedText;
                        scrollToBottom();
                    }
                }
            } else {
                let errorText = 'Unknown error';
                try {
                    const data = await response.json();
                    errorText = data.error || errorText;
                } catch(e) {}
                appendMessage('Sorry, I encountered an error: ' + errorText, 'system');
            }
        } catch (error) {
            removeElement(loadingId);
            appendMessage('Network error. Please check your connection to the server.', 'system');
            console.error('Chat error:', error);
        }
    });
    
    ingestBtn.addEventListener('click', async () => {
        const confirmSync = confirm("This will trigger a scrape of OneWorld Technologies and update the Pinecone vector database. Do you want to proceed?");
        if (!confirmSync) return;
        
        appendMessage("Triggering data sync... this may take a minute.", 'system');
        const loadingId = appendLoadingIndicator();
        
        try {
            const response = await fetch('http://127.0.0.1:8000/api/ingest', { method: 'POST' });
            const data = await response.json();
            
            removeElement(loadingId);
            if (response.ok) {
                appendMessage("✅ " + data.status, 'system');
            } else {
                appendMessage("❌ Error during sync: " + (data.error || 'Unknown error'), 'system');
            }
        } catch (error) {
            removeElement(loadingId);
            appendMessage("❌ Network error during sync.", 'system');
            console.error('Sync error:', error);
        }
    });
    
    function appendMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = `avatar ${sender}-avatar`;
        avatarDiv.innerHTML = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Simple markdown parsing for bold and line breaks
        const formattedText = text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            
        contentDiv.innerHTML = `<p>${formattedText}</p>`;
        
        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(contentDiv);
        
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
    }
    
    function appendLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = `message system-message`;
        msgDiv.id = id;
        
        msgDiv.innerHTML = `
            <div class="avatar system-avatar">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }
    
    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) {
            el.remove();
        }
    }
    
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
