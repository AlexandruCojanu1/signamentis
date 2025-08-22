#!/usr/bin/env python3
"""
SignaMentis MCP Agent Manager

This module implements a Model Context Protocol (MCP) agent manager
to coordinate AI models and provide standardized communication interfaces.

Author: SignaMentis Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Standardized message format for agent communication."""
    sender: str
    recipient: str
    message_type: str
    content: Any
    timestamp: datetime
    priority: int = 1
    metadata: Optional[Dict] = None


@dataclass
class AgentCapability:
    """Agent capability definition."""
    name: str
    description: str
    input_schema: Dict
    output_schema: Dict
    async_support: bool = False
    batch_support: bool = False


class MCPAgentManager:
    """
    Model Context Protocol Agent Manager for SignaMentis.
    
    Coordinates AI models and provides standardized interfaces for:
    - Model communication and coordination
    - Data flow management
    - Task scheduling and execution
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the MCP Agent Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.agents = {}
        self.agent_capabilities = {}
        self.message_queue = asyncio.Queue()
        self.message_history = []
        self.performance_metrics = {}
        self.is_running = False
        
        # Database for storing results and metadata
        self.db_path = Path("data/mcp_agent_manager.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("MCP Agent Manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database for storing agent data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create agents table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS agents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        agent_type TEXT NOT NULL,
                        status TEXT DEFAULT 'active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        capabilities TEXT,
                        performance_metrics TEXT
                    )
                ''')
                
                # Create messages table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sender TEXT NOT NULL,
                        recipient TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        content TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        priority INTEGER DEFAULT 1,
                        metadata TEXT
                    )
                ''')
                
                # Create results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        result_data TEXT,
                        execution_time REAL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN DEFAULT 1,
                        error_message TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def register_agent(self, 
                      name: str, 
                      agent_type: str, 
                      capabilities: List[AgentCapability],
                      agent_instance: Any = None) -> bool:
        """
        Register a new agent with the manager.
        
        Args:
            name: Agent name
            agent_type: Type of agent (e.g., 'ai_model', 'data_processor', 'risk_manager')
            capabilities: List of agent capabilities
            agent_instance: Actual agent instance
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Store agent information
            self.agents[name] = {
                'type': agent_type,
                'instance': agent_instance,
                'capabilities': capabilities,
                'status': 'active',
                'created_at': datetime.now(),
                'last_active': datetime.now()
            }
            
            # Store capabilities
            self.agent_capabilities[name] = capabilities
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO agents 
                    (name, agent_type, capabilities, last_active)
                    VALUES (?, ?, ?, ?)
                ''', (
                    name, 
                    agent_type, 
                    json.dumps([asdict(cap) for cap in capabilities]),
                    datetime.now()
                ))
                conn.commit()
            
            logger.info(f"Agent '{name}' registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering agent '{name}': {e}")
            return False
    
    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent."""
        try:
            if name in self.agents:
                del self.agents[name]
                if name in self.agent_capabilities:
                    del self.agent_capabilities[name]
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('UPDATE agents SET status = ? WHERE name = ?', ('inactive', name))
                    conn.commit()
                
                logger.info(f"Agent '{name}' unregistered")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error unregistering agent '{name}': {e}")
            return False
    
    async def send_message(self, message: AgentMessage) -> bool:
        """
        Send a message to an agent.
        
        Args:
            message: AgentMessage object
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            # Validate recipient
            if message.recipient not in self.agents:
                logger.error(f"Recipient '{message.recipient}' not found")
                return False
            
            # Add to message queue
            await self.message_queue.put(message)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO messages 
                    (sender, recipient, message_type, content, timestamp, priority, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message.sender,
                    message.recipient,
                    message.message_type,
                    json.dumps(message.content),
                    message.timestamp,
                    message.priority,
                    json.dumps(message.metadata) if message.metadata else None
                ))
                conn.commit()
            
            # Update agent last active time
            self.agents[message.recipient]['last_active'] = datetime.now()
            
            logger.debug(f"Message sent from '{message.sender}' to '{message.recipient}'")
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def broadcast_message(self, 
                              sender: str, 
                              message_type: str, 
                              content: Any,
                              exclude_sender: bool = True) -> int:
        """
        Broadcast a message to all agents.
        
        Args:
            sender: Sender agent name
            message_type: Type of message
            content: Message content
            exclude_sender: Whether to exclude sender from recipients
            
        Returns:
            int: Number of messages sent
        """
        sent_count = 0
        recipients = [name for name in self.agents.keys() if not exclude_sender or name != sender]
        
        for recipient in recipients:
            message = AgentMessage(
                sender=sender,
                recipient=recipient,
                message_type=message_type,
                content=content,
                timestamp=datetime.now()
            )
            
            if await self.send_message(message):
                sent_count += 1
        
        logger.info(f"Broadcast message sent to {sent_count} agents")
        return sent_count
    
    async def get_agent_status(self, agent_name: str) -> Optional[Dict]:
        """Get status of a specific agent."""
        if agent_name not in self.agents:
            return None
        
        agent = self.agents[agent_name]
        return {
            'name': agent_name,
            'type': agent['type'],
            'status': agent['status'],
            'capabilities': [asdict(cap) for cap in agent['capabilities']],
            'created_at': agent['created_at'],
            'last_active': agent['last_active']
        }
    
    async def get_all_agents_status(self) -> Dict[str, Dict]:
        """Get status of all agents."""
        status = {}
        for name in self.agents:
            status[name] = await self.get_agent_status(name)
        return status
    
    async def execute_task(self, 
                          task_type: str, 
                          agent_name: str, 
                          task_data: Any,
                          timeout: float = 30.0) -> Optional[Dict]:
        """
        Execute a task on a specific agent.
        
        Args:
            task_type: Type of task to execute
            agent_name: Name of the agent to execute task on
            task_data: Data for the task
            timeout: Task timeout in seconds
            
        Returns:
            Dict: Task result or None if failed
        """
        if agent_name not in self.agents:
            logger.error(f"Agent '{agent_name}' not found")
            return None
        
        try:
            start_time = time.time()
            
            # Find agent capability for this task
            agent_caps = self.agent_capabilities.get(agent_name, [])
            task_capability = None
            
            for cap in agent_caps:
                if cap.name == task_type:
                    task_capability = cap
                    break
            
            if not task_capability:
                logger.error(f"Agent '{agent_name}' doesn't support task '{task_type}'")
                return None
            
            # Execute task
            agent_instance = self.agents[agent_name]['instance']
            
            if task_capability.async_support:
                # Async execution
                if hasattr(agent_instance, task_type):
                    task_method = getattr(agent_instance, task_type)
                    if asyncio.iscoroutinefunction(task_method):
                        result = await asyncio.wait_for(task_method(task_data), timeout=timeout)
                    else:
                        # Run sync method in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.executor, task_method, task_data
                        )
                else:
                    logger.error(f"Task method '{task_type}' not found on agent '{agent_name}'")
                    return None
            else:
                # Sync execution
                if hasattr(agent_instance, task_type):
                    task_method = getattr(agent_instance, task_type)
                    # Run in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, task_method, task_data
                    )
                else:
                    logger.error(f"Task method '{task_type}' not found on agent '{agent_name}'")
                    return None
            
            execution_time = time.time() - start_time
            
            # Store result in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO results 
                    (agent_name, task_type, result_data, execution_time, timestamp, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    agent_name,
                    task_type,
                    json.dumps(result),
                    execution_time,
                    datetime.now(),
                    True
                ))
                conn.commit()
            
            # Update performance metrics
            if agent_name not in self.performance_metrics:
                self.performance_metrics[agent_name] = {}
            
            if task_type not in self.performance_metrics[agent_name]:
                self.performance_metrics[agent_name][task_type] = []
            
            self.performance_metrics[agent_name][task_type].append({
                'execution_time': execution_time,
                'timestamp': datetime.now(),
                'success': True
            })
            
            logger.info(f"Task '{task_type}' executed successfully on '{agent_name}' in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Task '{task_type}' on '{agent_name}' timed out after {timeout}s")
            self._record_task_failure(agent_name, task_type, "Timeout", time.time() - start_time)
            return None
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing task '{task_type}' on '{agent_name}': {e}")
            self._record_task_failure(agent_name, task_type, str(e), execution_time)
            return None
    
    def _record_task_failure(self, agent_name: str, task_type: str, error_message: str, execution_time: float):
        """Record task failure in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO results 
                    (agent_name, task_type, result_data, execution_time, timestamp, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    agent_name,
                    task_type,
                    None,
                    execution_time,
                    datetime.now(),
                    False,
                    error_message
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error recording task failure: {e}")
    
    async def start(self):
        """Start the MCP Agent Manager."""
        if self.is_running:
            logger.warning("MCP Agent Manager is already running")
            return
        
        self.is_running = True
        logger.info("Starting MCP Agent Manager...")
        
        # Start message processing loop
        asyncio.create_task(self._message_processor())
        
        logger.info("MCP Agent Manager started successfully")
    
    async def stop(self):
        """Stop the MCP Agent Manager."""
        if not self.is_running:
            logger.warning("MCP Agent Manager is not running")
            return
        
        self.is_running = False
        logger.info("Stopping MCP Agent Manager...")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("MCP Agent Manager stopped")
    
    async def _message_processor(self):
        """Process messages from the queue."""
        while self.is_running:
            try:
                # Get message from queue with timeout
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process message
                await self._process_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                continue
    
    async def _process_message(self, message: AgentMessage):
        """Process a single message."""
        try:
            # Update message history
            self.message_history.append(message)
            
            # Keep only last 1000 messages
            if len(self.message_history) > 1000:
                self.message_history = self.message_history[-1000:]
            
            logger.debug(f"Processing message: {message.sender} -> {message.recipient}: {message.message_type}")
            
            # Here you can add custom message processing logic
            # For now, just log the message
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for all agents."""
        summary = {}
        
        for agent_name, metrics in self.performance_metrics.items():
            summary[agent_name] = {}
            
            for task_type, task_metrics in metrics.items():
                if task_metrics:
                    execution_times = [m['execution_time'] for m in task_metrics if m['success']]
                    success_count = sum(1 for m in task_metrics if m['success'])
                    total_count = len(task_metrics)
                    
                    summary[agent_name][task_type] = {
                        'total_executions': total_count,
                        'successful_executions': success_count,
                        'success_rate': success_count / total_count if total_count > 0 else 0,
                        'avg_execution_time': np.mean(execution_times) if execution_times else 0,
                        'min_execution_time': np.min(execution_times) if execution_times else 0,
                        'max_execution_time': np.max(execution_times) if execution_times else 0
                    }
        
        return summary


def create_mcp_agent_manager(config: Optional[Dict] = None) -> MCPAgentManager:
    """
    Create MCP Agent Manager instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MCPAgentManager instance
    """
    return MCPAgentManager(config)


if __name__ == "__main__":
    # Test the MCP Agent Manager
    async def test_mcp_manager():
        manager = create_mcp_agent_manager()
        
        # Start manager
        await manager.start()
        
        # Create test agent capabilities
        test_capabilities = [
            AgentCapability(
                name="predict",
                description="Make price predictions",
                input_schema={"features": "array", "timeframe": "string"},
                output_schema={"prediction": "float", "confidence": "float"},
                async_support=True
            )
        ]
        
        # Register test agent
        class TestAgent:
            async def predict(self, data):
                await asyncio.sleep(0.1)  # Simulate work
                return {"prediction": 2000.0, "confidence": 0.8}
        
        test_agent = TestAgent()
        manager.register_agent("test_agent", "ai_model", test_capabilities, test_agent)
        
        # Test message sending
        message = AgentMessage(
            sender="system",
            recipient="test_agent",
            message_type="predict",
            content={"features": [1, 2, 3], "timeframe": "15m"},
            timestamp=datetime.now()
        )
        
        await manager.send_message(message)
        
        # Test task execution
        result = await manager.execute_task("predict", "test_agent", {"features": [1, 2, 3]})
        print(f"Task result: {result}")
        
        # Get status
        status = await manager.get_all_agents_status()
        print(f"Agent status: {status}")
        
        # Stop manager
        await manager.stop()
    
    # Run test
    asyncio.run(test_mcp_manager())

