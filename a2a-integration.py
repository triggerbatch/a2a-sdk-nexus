# nexus/nexus_base/a2a_manager.py
"""
Agent-to-Agent (A2A) Protocol Manager for Nexus
Provides integration with Google's A2A protocol for multi-agent communication
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict

# Import A2A SDK components (assuming a2a-python is installed)
try:
    from a2a import (
        AgentCard,
        AgentExecutor,
        A2AServer,
        A2AClient,
        AgentDiscovery,
        StreamResponse,
        AgentNetwork,
        Message as A2AMessage,
        AgentCapability,
        AgentEndpoint
    )
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    logging.warning("A2A SDK not available. Install with: pip install a2a-python")

from nexus.nexus_base.agent_manager import BaseAgent
from nexus.nexus_base.nexus_models import Message, Thread, ChatParticipants, db


@dataclass
class A2AAgentCard:
    """Agent Card for A2A protocol discovery and capabilities"""
    id: str
    name: str
    description: str
    version: str
    capabilities: List[str]
    endpoints: Dict[str, str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_nexus_agent(cls, agent: BaseAgent, base_url: str = "http://localhost:8080"):
        """Create an A2A Agent Card from a Nexus agent"""
        capabilities = []
        if agent.supports_actions:
            capabilities.append("actions")
        if agent.supports_memory:
            capabilities.append("memory")
        if agent.supports_knowledge:
            capabilities.append("knowledge")
        
        return cls(
            id=f"nexus-{agent.name}-{uuid.uuid4().hex[:8]}",
            name=agent.name,
            description=f"Nexus Agent: {agent.name}",
            version="1.0.0",
            capabilities=capabilities,
            endpoints={
                "chat": f"{base_url}/a2a/agents/{agent.name}/chat",
                "stream": f"{base_url}/a2a/agents/{agent.name}/stream",
                "actions": f"{base_url}/a2a/agents/{agent.name}/actions",
            },
            metadata={
                "profile": agent.profile.name if agent.profile else None,
                "actions": [a["name"] for a in agent.actions] if agent.actions else [],
            },
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )


class A2AAgentExecutor:
    """Executor for A2A protocol agent operations"""
    
    def __init__(self, agent: BaseAgent, nexus):
        self.agent = agent
        self.nexus = nexus
        self.execution_history = []
        
    async def execute(self, message: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute an A2A protocol message"""
        try:
            msg_type = message.get("type", "chat")
            content = message.get("content", "")
            
            if msg_type == "chat":
                response = await self._handle_chat(content, context)
            elif msg_type == "action":
                response = await self._handle_action(message, context)
            elif msg_type == "query":
                response = await self._handle_query(message, context)
            else:
                response = {"error": f"Unknown message type: {msg_type}"}
            
            # Log execution
            self.execution_history.append({
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "response": response,
                "context": context
            })
            
            return response
            
        except Exception as e:
            logging.error(f"A2A Executor error: {e}")
            return {"error": str(e)}
    
    async def _handle_chat(self, content: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Handle chat messages"""
        thread_id = context.get("thread_id") if context else None
        
        # Apply RAG if configured
        if hasattr(self.agent, 'knowledge_store') and self.agent.knowledge_store:
            knowledge_rag = self.nexus.apply_knowledge_RAG(
                self.agent.knowledge_store, content
            )
            content = content + knowledge_rag
        
        if hasattr(self.agent, 'memory_store') and self.agent.memory_store:
            memory_rag = self.nexus.apply_memory_RAG(
                self.agent.memory_store, content, self.agent
            )
            content = content + memory_rag
        
        # Get response from agent
        response = await self.agent.get_response(content, thread_id)
        
        return {
            "type": "chat_response",
            "content": response,
            "agent": self.agent.name,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_action(self, message: Dict, context: Optional[Dict]) -> Dict[str, Any]:
        """Handle action execution requests"""
        action_name = message.get("action")
        parameters = message.get("parameters", {})
        
        # Find and execute the action
        for action in self.agent.actions:
            if action["name"] == action_name:
                function = action["pointer"]
                result = function(**parameters, _caller_agent=self.agent)
                return {
                    "type": "action_response",
                    "action": action_name,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
        
        return {"error": f"Action {action_name} not found"}
    
    async def _handle_query(self, message: Dict, context: Optional[Dict]) -> Dict[str, Any]:
        """Handle query requests"""
        query_type = message.get("query_type", "capabilities")
        
        if query_type == "capabilities":
            return {
                "type": "query_response",
                "capabilities": {
                    "actions": self.agent.supports_actions,
                    "memory": self.agent.supports_memory,
                    "knowledge": self.agent.supports_knowledge,
                    "available_actions": [a["name"] for a in self.agent.actions] if self.agent.actions else []
                }
            }
        
        return {"error": f"Unknown query type: {query_type}"}


class A2AStreamHandler:
    """Handler for streaming responses in A2A protocol"""
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.active_streams = {}
    
    async def create_stream(self, content: str, stream_id: Optional[str] = None) -> str:
        """Create a new stream for agent responses"""
        if not stream_id:
            stream_id = str(uuid.uuid4())
        
        self.active_streams[stream_id] = {
            "content": content,
            "created_at": datetime.now(),
            "chunks": []
        }
        
        return stream_id
    
    async def stream_response(self, stream_id: str, thread_id: Optional[str] = None):
        """Stream response chunks"""
        if stream_id not in self.active_streams:
            yield {"error": "Stream not found"}
            return
        
        stream_info = self.active_streams[stream_id]
        content = stream_info["content"]
        
        # Get streaming response from agent
        if hasattr(self.agent, 'get_response_stream'):
            response_generator = self.agent.get_response_stream(content, thread_id)
            
            for chunk in response_generator():
                stream_info["chunks"].append(chunk)
                yield {
                    "type": "stream_chunk",
                    "stream_id": stream_id,
                    "chunk": chunk,
                    "timestamp": datetime.now().isoformat()
                }
        
        # Clean up stream
        del self.active_streams[stream_id]
        
        yield {
            "type": "stream_complete",
            "stream_id": stream_id,
            "timestamp": datetime.now().isoformat()
        }


class A2ADiscoveryService:
    """Service for discovering and registering A2A agents"""
    
    def __init__(self, nexus):
        self.nexus = nexus
        self.registered_agents = {}
        self.discovered_agents = {}
    
    async def register_agent(self, agent: BaseAgent, base_url: str = "http://localhost:8080") -> A2AAgentCard:
        """Register a Nexus agent for A2A discovery"""
        card = A2AAgentCard.from_nexus_agent(agent, base_url)
        self.registered_agents[card.id] = {
            "card": card,
            "agent": agent,
            "registered_at": datetime.now()
        }
        
        logging.info(f"Registered A2A agent: {card.id}")
        return card
    
    async def discover_agents(self, query: Optional[Dict] = None) -> List[A2AAgentCard]:
        """Discover available A2A agents"""
        agents = []
        
        # Add local registered agents
        for agent_info in self.registered_agents.values():
            card = agent_info["card"]
            if self._matches_query(card, query):
                agents.append(card)
        
        # Add discovered external agents
        for card in self.discovered_agents.values():
            if self._matches_query(card, query):
                agents.append(card)
        
        return agents
    
    def _matches_query(self, card: A2AAgentCard, query: Optional[Dict]) -> bool:
        """Check if agent card matches discovery query"""
        if not query:
            return True
        
        # Check capabilities
        if "capabilities" in query:
            required_caps = set(query["capabilities"])
            agent_caps = set(card.capabilities)
            if not required_caps.issubset(agent_caps):
                return False
        
        # Check name pattern
        if "name_pattern" in query:
            import re
            if not re.match(query["name_pattern"], card.name):
                return False
        
        return True
    
    async def add_external_agent(self, agent_card: Dict) -> None:
        """Add an externally discovered agent"""
        card = A2AAgentCard(**agent_card)
        self.discovered_agents[card.id] = card
        logging.info(f"Added external A2A agent: {card.id}")


class A2ANetworkManager:
    """Manager for multi-agent A2A network operations"""
    
    def __init__(self, nexus):
        self.nexus = nexus
        self.networks = {}
        self.network_agents = {}
    
    async def create_network(self, network_id: str, agents: List[str]) -> Dict:
        """Create a new A2A agent network"""
        network = {
            "id": network_id,
            "agents": agents,
            "created_at": datetime.now(),
            "messages": [],
            "state": "active"
        }
        
        self.networks[network_id] = network
        
        # Track agent participation
        for agent_name in agents:
            if agent_name not in self.network_agents:
                self.network_agents[agent_name] = []
            self.network_agents[agent_name].append(network_id)
        
        return network
    
    async def broadcast_message(self, network_id: str, sender: str, message: Dict) -> List[Dict]:
        """Broadcast a message to all agents in network"""
        if network_id not in self.networks:
            return [{"error": "Network not found"}]
        
        network = self.networks[network_id]
        responses = []
        
        for agent_name in network["agents"]:
            if agent_name != sender:
                agent = self.nexus.get_agent(agent_name)
                if agent:
                    executor = A2AAgentExecutor(agent, self.nexus)
                    response = await executor.execute(message, {"network_id": network_id})
                    responses.append({
                        "agent": agent_name,
                        "response": response
                    })
        
        # Log the broadcast
        network["messages"].append({
            "sender": sender,
            "message": message,
            "responses": responses,
            "timestamp": datetime.now().isoformat()
        })
        
        return responses
    
    async def orchestrate_workflow(self, network_id: str, workflow: Dict) -> Dict:
        """Orchestrate a workflow across network agents"""
        if network_id not in self.networks:
            return {"error": "Network not found"}
        
        results = {}
        steps = workflow.get("steps", [])
        
        for step in steps:
            agent_name = step.get("agent")
            message = step.get("message")
            depends_on = step.get("depends_on", [])
            
            # Wait for dependencies
            for dep in depends_on:
                if dep not in results:
                    return {"error": f"Dependency {dep} not satisfied"}
            
            # Execute step
            agent = self.nexus.get_agent(agent_name)
            if agent:
                executor = A2AAgentExecutor(agent, self.nexus)
                context = {
                    "network_id": network_id,
                    "previous_results": {k: results[k] for k in depends_on}
                }
                result = await executor.execute(message, context)
                results[step.get("id", agent_name)] = result
        
        return {
            "workflow_id": workflow.get("id"),
            "network_id": network_id,
            "results": results,
            "completed_at": datetime.now().isoformat()
        }


class A2AManager:
    """Main A2A Protocol Manager for Nexus"""
    
    def __init__(self, nexus):
        self.nexus = nexus
        self.discovery = A2ADiscoveryService(nexus)
        self.network_manager = A2ANetworkManager(nexus)
        self.executors = {}
        self.stream_handlers = {}
        self.server = None
        self.client = None
        
        # Initialize A2A components if available
        if A2A_AVAILABLE:
            self._init_a2a_components()
    
    def _init_a2a_components(self):
        """Initialize A2A SDK components"""
        try:
            self.server = A2AServer()
            self.client = A2AClient()
            logging.info("A2A SDK components initialized")
        except Exception as e:
            logging.error(f"Failed to initialize A2A SDK: {e}")
    
    async def register_all_agents(self, base_url: str = "http://localhost:8080"):
        """Register all Nexus agents for A2A discovery"""
        agent_names = self.nexus.get_agent_names()
        cards = []
        
        for agent_name in agent_names:
            agent = self.nexus.get_agent(agent_name)
            if agent:
                card = await self.discovery.register_agent(agent, base_url)
                cards.append(card)
                
                # Create executor and stream handler
                self.executors[agent_name] = A2AAgentExecutor(agent, self.nexus)
                self.stream_handlers[agent_name] = A2AStreamHandler(agent)
        
        return cards
    
    async def handle_a2a_request(self, agent_name: str, request: Dict) -> Dict:
        """Handle incoming A2A protocol request"""
        if agent_name not in self.executors:
            agent = self.nexus.get_agent(agent_name)
            if not agent:
                return {"error": f"Agent {agent_name} not found"}
            self.executors[agent_name] = A2AAgentExecutor(agent, self.nexus)
        
        executor = self.executors[agent_name]
        return await executor.execute(request)
    
    async def stream_agent_response(self, agent_name: str, content: str, stream_id: Optional[str] = None):
        """Stream agent response using A2A protocol"""
        if agent_name not in self.stream_handlers:
            agent = self.nexus.get_agent(agent_name)
            if not agent:
                yield {"error": f"Agent {agent_name} not found"}
                return
            self.stream_handlers[agent_name] = A2AStreamHandler(agent)
        
        handler = self.stream_handlers[agent_name]
        stream_id = await handler.create_stream(content, stream_id)
        
        async for chunk in handler.stream_response(stream_id):
            yield chunk
    
    async def create_agent_network(self, agents: List[str], network_id: Optional[str] = None) -> Dict:
        """Create a multi-agent network"""
        if not network_id:
            network_id = f"network-{uuid.uuid4().hex[:8]}"
        
        return await self.network_manager.create_network(network_id, agents)
    
    async def send_network_message(self, network_id: str, sender: str, message: Dict) -> List[Dict]:
        """Send message to agent network"""
        return await self.network_manager.broadcast_message(network_id, sender, message)
    
    async def run_network_workflow(self, network_id: str, workflow: Dict) -> Dict:
        """Run a workflow across agent network"""
        return await self.network_manager.orchestrate_workflow(network_id, workflow)
    
    def get_agent_card(self, agent_name: str) -> Optional[A2AAgentCard]:
        """Get A2A agent card for a Nexus agent"""
        for agent_info in self.discovery.registered_agents.values():
            if agent_info["agent"].name == agent_name:
                return agent_info["card"]
        return None
    
    async def connect_to_external_agent(self, agent_url: str, agent_card: Dict) -> bool:
        """Connect to an external A2A agent"""
        try:
            await self.discovery.add_external_agent(agent_card)
            if self.client and A2A_AVAILABLE:
                # Use A2A SDK client to establish connection
                await self.client.connect(agent_url)
            return True
        except Exception as e:
            logging.error(f"Failed to connect to external agent: {e}")
            return False
    
    async def start_a2a_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start A2A protocol server"""
        if self.server and A2A_AVAILABLE:
            try:
                await self.server.start(host, port)
                logging.info(f"A2A server started on {host}:{port}")
            except Exception as e:
                logging.error(f"Failed to start A2A server: {e}")
        else:
            logging.warning("A2A server not available. Using fallback HTTP endpoints.")
    
    async def stop_a2a_server(self):
        """Stop A2A protocol server"""
        if self.server and A2A_AVAILABLE:
            await self.server.stop()
            logging.info("A2A server stopped")


# Extension to existing Nexus class
def extend_nexus_with_a2a(nexus_instance):
    """Extend an existing Nexus instance with A2A capabilities"""
    nexus_instance.a2a_manager = A2AManager(nexus_instance)
    
    # Add A2A methods to Nexus instance
    nexus_instance.register_a2a_agents = nexus_instance.a2a_manager.register_all_agents
    nexus_instance.handle_a2a_request = nexus_instance.a2a_manager.handle_a2a_request
    nexus_instance.stream_a2a_response = nexus_instance.a2a_manager.stream_agent_response
    nexus_instance.create_agent_network = nexus_instance.a2a_manager.create_agent_network
    nexus_instance.send_network_message = nexus_instance.a2a_manager.send_network_message
    nexus_instance.run_network_workflow = nexus_instance.a2a_manager.run_network_workflow
    nexus_instance.get_agent_card = nexus_instance.a2a_manager.get_agent_card
    nexus_instance.connect_external_agent = nexus_instance.a2a_manager.connect_to_external_agent
    nexus_instance.start_a2a_server = nexus_instance.a2a_manager.start_a2a_server
    nexus_instance.stop_a2a_server = nexus_instance.a2a_manager.stop_a2a_server
    nexus_instance.discover_a2a_agents = nexus_instance.a2a_manager.discovery.discover_agents
    
    logging.info("Nexus extended with A2A protocol support")
    return nexus_instance
