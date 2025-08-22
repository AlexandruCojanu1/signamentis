#!/usr/bin/env python3
"""
SignaMentis - Redis Message Bus

This module implements a Redis-based message bus for inter-service communication
with standardized message envelopes and pub/sub patterns.

Author: SignaMentis Team
Version: 2.0.0
"""

import redis
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid
import pickle
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages that can be sent through the bus."""
    TRADE_SIGNAL = "trade_signal"
    RISK_UPDATE = "risk_update"
    NEWS_UPDATE = "news_update"
    MODEL_UPDATE = "model_update"
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_METRIC = "performance_metric"
    DATA_UPDATE = "data_update"
    EXECUTION_UPDATE = "execution_update"
    HEARTBEAT = "heartbeat"
    COMMAND = "command"

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class MessageEnvelope:
    """Standardized message envelope for all communications."""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    timestamp: datetime
    source: str
    destination: Optional[str]
    correlation_id: Optional[str]
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageEnvelope':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        data['status'] = MessageStatus(data['status'])
        return cls(**data)

class RedisMessageBus:
    """
    Redis-based message bus for inter-service communication.
    
    Supports:
    - Pub/Sub messaging
    - Message queuing with priorities
    - Message persistence
    - Retry mechanisms
    - Message correlation
    - Dead letter queues
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30
    ):
        """
        Initialize Redis message bus.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout
            socket_connect_timeout: Socket connection timeout
            retry_on_timeout: Whether to retry on timeout
            health_check_interval: Health check interval in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        
        # Connection pool
        self.pool = None
        self.redis_client = None
        
        # Subscriptions
        self.subscriptions = {}
        self.message_handlers = {}
        
        # Health monitoring
        self.is_healthy = False
        self.last_health_check = None
        
        # Initialize connection
        self._connect()
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            self.pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
                decode_responses=True
            )
            
            self.redis_client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.redis_client.ping()
            self.is_healthy = True
            self.last_health_check = datetime.now()
            
            logger.info(f"✅ Connected to Redis at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.is_healthy = False
            raise
    
    def _ensure_connection(self):
        """Ensure Redis connection is active."""
        if not self.is_healthy:
            try:
                self.redis_client.ping()
                self.is_healthy = True
                self.last_health_check = datetime.now()
            except:
                self._connect()
    
    def health_check(self) -> bool:
        """Perform health check."""
        try:
            if self.redis_client:
                self.redis_client.ping()
                self.is_healthy = True
                self.last_health_check = datetime.now()
                return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self.is_healthy = False
        
        return False
    
    def publish(
        self,
        channel: str,
        message: Union[MessageEnvelope, Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """
        Publish message to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            priority: Message priority
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            
            # Convert to message envelope if needed
            if isinstance(message, dict):
                message = MessageEnvelope(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.SYSTEM_ALERT,
                    priority=priority,
                    timestamp=datetime.now(),
                    source="system",
                    destination=None,
                    correlation_id=None,
                    payload=message,
                    metadata={}
                )
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            # Publish to channel
            result = self.redis_client.publish(channel, message_data)
            
            # Store in priority queue for persistence
            self._store_message(channel, message, priority)
            
            logger.debug(f"📤 Published message {message.message_id} to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish message: {e}")
            return False
    
    def _store_message(self, channel: str, message: MessageEnvelope, priority: MessagePriority):
        """Store message in priority queue."""
        try:
            queue_key = f"queue:{channel}:{priority.value}"
            message_data = json.dumps(message.to_dict())
            
            # Add to sorted set with timestamp as score for ordering
            self.redis_client.zadd(queue_key, {message_data: datetime.now().timestamp()})
            
            # Set TTL for message cleanup
            if message.ttl:
                self.redis_client.expire(queue_key, message.ttl)
                
        except Exception as e:
            logger.warning(f"Failed to store message: {e}")
    
    def subscribe(
        self,
        channel: str,
        handler: Callable[[MessageEnvelope], None],
        pattern: bool = False
    ) -> bool:
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel name or pattern
            handler: Message handler function
            pattern: Whether this is a pattern subscription
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            
            # Store handler
            self.message_handlers[channel] = handler
            
            # Subscribe to channel
            pubsub = self.redis_client.pubsub()
            
            if pattern:
                pubsub.psubscribe(channel)
            else:
                pubsub.subscribe(channel)
            
            self.subscriptions[channel] = pubsub
            
            # Start listener in background
            asyncio.create_task(self._listen_for_messages(channel, pubsub))
            
            logger.info(f"📡 Subscribed to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to {channel}: {e}")
            return False
    
    async def _listen_for_messages(self, channel: str, pubsub):
        """Listen for messages on a channel."""
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        # Parse message
                        message_data = json.loads(message['data'])
                        envelope = MessageEnvelope.from_dict(message_data)
                        
                        # Call handler
                        handler = self.message_handlers.get(channel)
                        if handler:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(envelope)
                            else:
                                handler(envelope)
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing message: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Error listening to {channel}: {e}")
    
    def unsubscribe(self, channel: str) -> bool:
        """Unsubscribe from a channel."""
        try:
            if channel in self.subscriptions:
                self.subscriptions[channel].unsubscribe()
                del self.subscriptions[channel]
                del self.message_handlers[channel]
                logger.info(f"📡 Unsubscribed from {channel}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to unsubscribe from {channel}: {e}")
        
        return False
    
    def send_message(
        self,
        destination: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        source: str = "system",
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a direct message to a destination.
        
        Args:
            destination: Destination service/channel
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            source: Source service
            correlation_id: Correlation ID for tracking
            metadata: Additional metadata
            
        Returns:
            Message ID
        """
        message = MessageEnvelope(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            priority=priority,
            timestamp=datetime.now(),
            source=source,
            destination=destination,
            correlation_id=correlation_id,
            payload=payload,
            metadata=metadata or {}
        )
        
        # Publish to destination channel
        channel = f"direct:{destination}"
        success = self.publish(channel, message, priority)
        
        if success:
            return message.message_id
        else:
            raise RuntimeError("Failed to send message")
    
    def request_response(
        self,
        destination: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.NORMAL,
        source: str = "system"
    ) -> Optional[MessageEnvelope]:
        """
        Send a request and wait for response.
        
        Args:
            destination: Destination service
            message_type: Type of message
            payload: Message payload
            timeout: Response timeout
            priority: Message priority
            source: Source service
            
        Returns:
            Response message or None if timeout
        """
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        
        # Create request message
        request = MessageEnvelope(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            priority=priority,
            timestamp=datetime.now(),
            source=source,
            destination=destination,
            correlation_id=correlation_id,
            payload=payload,
            metadata={'request': True}
        )
        
        # Subscribe to response channel
        response_channel = f"response:{correlation_id}"
        response_received = asyncio.Event()
        response_message = None
        
        def response_handler(message: MessageEnvelope):
            if message.correlation_id == correlation_id:
                nonlocal response_message
                response_message = message
                response_received.set()
        
        self.subscribe(response_channel, response_handler)
        
        # Send request
        self.send_message(destination, message_type, payload, priority, source, correlation_id)
        
        # Wait for response
        try:
            asyncio.wait_for(response_received.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for {correlation_id}")
            return None
        finally:
            self.unsubscribe(response_channel)
        
        return response_message
    
    def get_queue_stats(self, channel: str) -> Dict[str, Any]:
        """Get statistics for a message queue."""
        try:
            self._ensure_connection()
            
            stats = {}
            
            for priority in MessagePriority:
                queue_key = f"queue:{channel}:{priority.value}"
                
                # Get queue length
                queue_length = self.redis_client.zcard(queue_key)
                
                # Get oldest and newest messages
                oldest = self.redis_client.zrange(queue_key, 0, 0, withscores=True)
                newest = self.redis_client.zrange(queue_key, -1, -1, withscores=True)
                
                stats[f"priority_{priority.value}"] = {
                    "length": queue_length,
                    "oldest_timestamp": oldest[0][1] if oldest else None,
                    "newest_timestamp": newest[0][1] if newest else None
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get queue stats: {e}")
            return {}
    
    def purge_queue(self, channel: str, priority: Optional[MessagePriority] = None) -> int:
        """Purge messages from a queue."""
        try:
            self._ensure_connection()
            
            if priority:
                queue_key = f"queue:{channel}:{priority.value}"
                return self.redis_client.delete(queue_key)
            else:
                # Purge all priorities
                count = 0
                for p in MessagePriority:
                    queue_key = f"queue:{channel}:{p.value}"
                    count += self.redis_client.delete(queue_key)
                return count
                
        except Exception as e:
            logger.error(f"❌ Failed to purge queue: {e}")
            return 0
    
    def close(self):
        """Close all connections and cleanup."""
        try:
            # Unsubscribe from all channels
            for channel in list(self.subscriptions.keys()):
                self.unsubscribe(channel)
            
            # Close Redis connections
            if self.pool:
                self.pool.disconnect()
            
            logger.info("🔌 Redis message bus closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing Redis bus: {e}")

class AsyncRedisMessageBus(RedisMessageBus):
    """Async version of Redis message bus."""
    
    async def publish_async(
        self,
        channel: str,
        message: Union[MessageEnvelope, Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """Async publish message."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.publish, channel, message, priority
        )
    
    async def send_message_async(
        self,
        destination: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        source: str = "system",
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Async send message."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.send_message, destination, message_type, payload,
            priority, source, correlation_id, metadata
        )

@asynccontextmanager
async def get_redis_bus(
    host: str = 'localhost',
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None
):
    """Async context manager for Redis message bus."""
    bus = AsyncRedisMessageBus(host=host, port=port, db=db, password=password)
    try:
        yield bus
    finally:
        bus.close()

# Example usage
if __name__ == "__main__":
    async def main():
        # Create message bus
        bus = RedisMessageBus()
        
        # Message handler
        def handle_message(message: MessageEnvelope):
            print(f"📨 Received: {message.message_type.value} from {message.source}")
            print(f"   Payload: {message.payload}")
        
        # Subscribe to channel
        bus.subscribe("test_channel", handle_message)
        
        # Publish message
        message = MessageEnvelope(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SYSTEM_ALERT,
            priority=MessagePriority.HIGH,
            timestamp=datetime.now(),
            source="test_service",
            destination=None,
            correlation_id=None,
            payload={"test": "data"},
            metadata={}
        )
        
        bus.publish("test_channel", message)
        
        # Wait a bit for message processing
        await asyncio.sleep(1)
        
        # Close bus
        bus.close()
    
    # Run example
    asyncio.run(main())
