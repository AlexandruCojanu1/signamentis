#!/usr/bin/env python3
"""
SignaMentis - RabbitMQ Message Bus

This module implements a RabbitMQ-based message bus for inter-service communication
with standardized message envelopes, routing, and queue management.

Author: SignaMentis Team
Version: 2.0.0
"""

import pika
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid
import threading
import time
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """RabbitMQ exchange types."""
    DIRECT = "direct"
    FANOUT = "fanout"
    TOPIC = "topic"
    HEADERS = "headers"

class RoutingKey(Enum):
    """Standard routing keys for message routing."""
    TRADE_SIGNAL = "trade.signal"
    RISK_UPDATE = "risk.update"
    NEWS_UPDATE = "news.update"
    MODEL_UPDATE = "model.update"
    SYSTEM_ALERT = "system.alert"
    PERFORMANCE_METRIC = "performance.metric"
    DATA_UPDATE = "data.update"
    EXECUTION_UPDATE = "execution.update"
    HEARTBEAT = "heartbeat"
    COMMAND = "command"

@dataclass
class QueueConfig:
    """Configuration for RabbitMQ queues."""
    name: str
    durable: bool = True
    auto_delete: bool = False
    exclusive: bool = False
    arguments: Optional[Dict[str, Any]] = None
    dead_letter_exchange: Optional[str] = None
    dead_letter_routing_key: Optional[str] = None
    message_ttl: Optional[int] = None
    max_length: Optional[int] = None
    max_bytes: Optional[int] = None

@dataclass
class ExchangeConfig:
    """Configuration for RabbitMQ exchanges."""
    name: str
    exchange_type: ExchangeType = ExchangeType.TOPIC
    durable: bool = True
    auto_delete: bool = False
    internal: bool = False
    arguments: Optional[Dict[str, Any]] = None

class RabbitMQMessageBus:
    """
    RabbitMQ-based message bus for inter-service communication.
    
    Supports:
    - Multiple exchange types (direct, fanout, topic, headers)
    - Message routing with routing keys
    - Queue management and configuration
    - Dead letter queues
    - Message persistence and durability
    - Connection pooling and recovery
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5672,
        username: str = 'guest',
        password: str = 'guest',
        virtual_host: str = '/',
        connection_pool_size: int = 5,
        heartbeat_interval: int = 600,
        blocked_connection_timeout: int = 300,
        connection_attempts: int = 3,
        retry_delay: float = 5.0
    ):
        """
        Initialize RabbitMQ message bus.
        
        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password
            virtual_host: RabbitMQ virtual host
            connection_pool_size: Connection pool size
            heartbeat_interval: Heartbeat interval in seconds
            blocked_connection_timeout: Blocked connection timeout
            connection_attempts: Number of connection attempts
            retry_delay: Delay between retry attempts
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.connection_pool_size = connection_pool_size
        self.heartbeat_interval = heartbeat_interval
        self.blocked_connection_timeout = blocked_connection_timeout
        self.connection_attempts = connection_attempts
        self.retry_delay = retry_delay
        
        # Connection management
        self.connection = None
        self.channel = None
        self.connection_pool = []
        
        # Queue and exchange configurations
        self.queues = {}
        self.exchanges = {}
        
        # Message handlers
        self.message_handlers = {}
        
        # Health monitoring
        self.is_healthy = False
        self.last_health_check = None
        
        # Initialize connection
        self._connect()
    
    def _connect(self):
        """Establish RabbitMQ connection."""
        try:
            # Connection parameters
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=credentials,
                heartbeat=self.heartbeat_interval,
                blocked_connection_timeout=self.blocked_connection_timeout,
                connection_attempts=self.connection_attempts,
                retry_delay=self.retry_delay
            )
            
            # Create connection
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Set QoS
            self.channel.basic_qos(prefetch_count=1)
            
            # Test connection
            self.channel.queue_declare(queue='health_check', durable=False, auto_delete=True)
            self.channel.queue_delete(queue='health_check')
            
            self.is_healthy = True
            self.last_health_check = datetime.now()
            
            logger.info(f"✅ Connected to RabbitMQ at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to RabbitMQ: {e}")
            self.is_healthy = False
            raise
    
    def _ensure_connection(self):
        """Ensure RabbitMQ connection is active."""
        if not self.is_healthy or not self.connection or self.connection.is_closed:
            try:
                self._connect()
            except Exception as e:
                logger.error(f"❌ Failed to reconnect: {e}")
                raise
    
    def health_check(self) -> bool:
        """Perform health check."""
        try:
            if self.connection and not self.connection.is_closed:
                # Declare a temporary queue to test connection
                self.channel.queue_declare(queue='health_check', durable=False, auto_delete=True)
                self.channel.queue_delete(queue='health_check')
                
                self.is_healthy = True
                self.last_health_check = datetime.now()
                return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self.is_healthy = False
        
        return False
    
    def declare_exchange(self, config: ExchangeConfig) -> bool:
        """
        Declare an exchange.
        
        Args:
            config: Exchange configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            
            self.channel.exchange_declare(
                exchange=config.name,
                exchange_type=config.exchange_type.value,
                durable=config.durable,
                auto_delete=config.auto_delete,
                internal=config.internal,
                arguments=config.arguments
            )
            
            self.exchanges[config.name] = config
            logger.info(f"✅ Declared exchange: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to declare exchange {config.name}: {e}")
            return False
    
    def declare_queue(self, config: QueueConfig) -> bool:
        """
        Declare a queue.
        
        Args:
            config: Queue configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            
            # Prepare arguments
            arguments = config.arguments or {}
            
            if config.dead_letter_exchange:
                arguments['x-dead-letter-exchange'] = config.dead_letter_exchange
            if config.dead_letter_routing_key:
                arguments['x-dead-letter-routing-key'] = config.dead_letter_routing_key
            if config.message_ttl:
                arguments['x-message-ttl'] = config.message_ttl
            if config.max_length:
                arguments['x-max-length'] = config.max_length
            if config.max_bytes:
                arguments['x-max-bytes'] = config.max_bytes
            
            self.channel.queue_declare(
                queue=config.name,
                durable=config.durable,
                exclusive=config.exclusive,
                auto_delete=config.auto_delete,
                arguments=arguments
            )
            
            self.queues[config.name] = config
            logger.info(f"✅ Declared queue: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to declare queue {config.name}: {e}")
            return False
    
    def bind_queue(
        self,
        queue_name: str,
        exchange_name: str,
        routing_key: str
    ) -> bool:
        """
        Bind a queue to an exchange with a routing key.
        
        Args:
            queue_name: Name of the queue
            exchange_name: Name of the exchange
            routing_key: Routing key for binding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            
            self.channel.queue_bind(
                queue=queue_name,
                exchange=exchange_name,
                routing_key=routing_key
            )
            
            logger.info(f"✅ Bound queue {queue_name} to {exchange_name} with key {routing_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to bind queue: {e}")
            return False
    
    def publish_message(
        self,
        exchange: str,
        routing_key: str,
        message: Union[Dict[str, Any], str],
        properties: Optional[pika.BasicProperties] = None,
        mandatory: bool = False
    ) -> bool:
        """
        Publish a message to an exchange.
        
        Args:
            exchange: Exchange name
            routing_key: Routing key
            message: Message to publish
            properties: Message properties
            mandatory: Whether message is mandatory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            
            # Prepare message
            if isinstance(message, dict):
                message_body = json.dumps(message).encode('utf-8')
            else:
                message_body = str(message).encode('utf-8')
            
            # Default properties
            if properties is None:
                properties = pika.BasicProperties(
                    delivery_mode=2,  # Persistent
                    content_type='application/json',
                    message_id=str(uuid.uuid4()),
                    timestamp=int(time.time()),
                    app_id='signa_mentis'
                )
            
            # Publish message
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=message_body,
                properties=properties,
                mandatory=mandatory
            )
            
            logger.debug(f"📤 Published message to {exchange} with key {routing_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish message: {e}")
            return False
    
    def consume_messages(
        self,
        queue_name: str,
        handler: Callable[[str, pika.BasicProperties, bytes], None],
        auto_ack: bool = False
    ) -> bool:
        """
        Start consuming messages from a queue.
        
        Args:
            queue_name: Name of the queue to consume from
            handler: Message handler function
            auto_ack: Whether to automatically acknowledge messages
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            
            # Store handler
            self.message_handlers[queue_name] = handler
            
            # Set up consumer
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=self._message_callback,
                auto_ack=auto_ack
            )
            
            logger.info(f"📡 Started consuming from queue: {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start consuming: {e}")
            return False
    
    def _message_callback(
        self,
        ch: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes
    ):
        """Internal message callback handler."""
        try:
            queue_name = method.routing_key or method.exchange
            
            # Get handler
            handler = self.message_handlers.get(queue_name)
            if handler:
                # Call handler
                handler(queue_name, properties, body)
                
                # Acknowledge message if not auto-ack
                if not ch.basic_ack:
                    ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                logger.warning(f"No handler found for queue: {queue_name}")
                # Reject message
                ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            # Reject message on error
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
    
    def start_consuming(self):
        """Start consuming messages (blocking)."""
        try:
            self._ensure_connection()
            logger.info("🚀 Starting message consumption...")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("⏹️  Stopping message consumption...")
            self.stop_consuming()
        except Exception as e:
            logger.error(f"❌ Error during consumption: {e}")
            self.stop_consuming()
    
    def stop_consuming(self):
        """Stop consuming messages."""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.stop_consuming()
            logger.info("⏹️  Message consumption stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping consumption: {e}")
    
    def get_queue_info(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Queue information or None if not found
        """
        try:
            self._ensure_connection()
            
            # Declare queue to get info
            method = self.channel.queue_declare(
                queue=queue_name,
                passive=True
            )
            
            return {
                'name': method.method.queue,
                'message_count': method.method.message_count,
                'consumer_count': method.method.consumer_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get queue info: {e}")
            return None
    
    def purge_queue(self, queue_name: str) -> int:
        """
        Purge all messages from a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Number of messages purged
        """
        try:
            self._ensure_connection()
            
            method = self.channel.queue_purge(queue=queue_name)
            count = method.method.message_count
            
            logger.info(f"🧹 Purged {count} messages from queue {queue_name}")
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to purge queue: {e}")
            return 0
    
    def delete_queue(self, queue_name: str, if_unused: bool = True, if_empty: bool = True) -> bool:
        """
        Delete a queue.
        
        Args:
            queue_name: Name of the queue
            if_unused: Only delete if unused
            if_empty: Only delete if empty
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            
            self.channel.queue_delete(
                queue=queue_name,
                if_unused=if_unused,
                if_empty=if_empty
            )
            
            if queue_name in self.queues:
                del self.queues[queue_name]
            
            logger.info(f"🗑️  Deleted queue: {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete queue: {e}")
            return False
    
    def close(self):
        """Close all connections and cleanup."""
        try:
            # Stop consuming
            self.stop_consuming()
            
            # Close channel and connection
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            
            logger.info("🔌 RabbitMQ message bus closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing RabbitMQ bus: {e}")

class AsyncRabbitMQMessageBus(RabbitMQMessageBus):
    """Async version of RabbitMQ message bus."""
    
    async def publish_message_async(
        self,
        exchange: str,
        routing_key: str,
        message: Union[Dict[str, Any], str],
        properties: Optional[pika.BasicProperties] = None,
        mandatory: bool = False
    ) -> bool:
        """Async publish message."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.publish_message, exchange, routing_key, message, properties, mandatory
        )
    
    async def start_consuming_async(self):
        """Start consuming messages asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.start_consuming)

@asynccontextmanager
async def get_rabbitmq_bus(
    host: str = 'localhost',
    port: int = 5672,
    username: str = 'guest',
    password: str = 'guest',
    virtual_host: str = '/'
):
    """Async context manager for RabbitMQ message bus."""
    bus = AsyncRabbitMQMessageBus(
        host=host, port=port, username=username, password=password, virtual_host=virtual_host
    )
    try:
        yield bus
    finally:
        bus.close()

def create_standard_exchanges() -> List[ExchangeConfig]:
    """Create standard exchange configurations."""
    return [
        ExchangeConfig(
            name="signa_mentis.trades",
            exchange_type=ExchangeType.TOPIC,
            durable=True
        ),
        ExchangeConfig(
            name="signa_mentis.risk",
            exchange_type=ExchangeType.TOPIC,
            durable=True
        ),
        ExchangeConfig(
            name="signa_mentis.news",
            exchange_type=ExchangeType.FANOUT,
            durable=True
        ),
        ExchangeConfig(
            name="signa_mentis.system",
            exchange_type=ExchangeType.DIRECT,
            durable=True
        ),
        ExchangeConfig(
            name="signa_mentis.dlq",  # Dead Letter Queue
            exchange_type=ExchangeType.DIRECT,
            durable=True
        )
    ]

def create_standard_queues() -> List[QueueConfig]:
    """Create standard queue configurations."""
    return [
        QueueConfig(
            name="trade_signals",
            durable=True,
            arguments={
                'x-dead-letter-exchange': 'signa_mentis.dlq',
                'x-dead-letter-routing-key': 'trade_signals.dlq'
            }
        ),
        QueueConfig(
            name="risk_updates",
            durable=True,
            arguments={
                'x-dead-letter-exchange': 'signa_mentis.dlq',
                'x-dead-letter-routing-key': 'risk_updates.dlq'
            }
        ),
        QueueConfig(
            name="news_updates",
            durable=True,
            arguments={
                'x-dead-letter-exchange': 'signa_mentis.dlq',
                'x-dead-letter-routing-key': 'news_updates.dlq'
            }
        ),
        QueueConfig(
            name="system_alerts",
            durable=True,
            arguments={
                'x-dead-letter-exchange': 'signa_mentis.dlq',
                'x-dead-letter-routing-key': 'system_alerts.dlq'
            }
        ),
        QueueConfig(
            name="dlq.trade_signals",
            durable=True
        ),
        QueueConfig(
            name="dlq.risk_updates",
            durable=True
        ),
        QueueConfig(
            name="dlq.news_updates",
            durable=True
        ),
        QueueConfig(
            name="dlq.system_alerts",
            durable=True
        )
    ]

# Example usage
if __name__ == "__main__":
    def main():
        # Create message bus
        bus = RabbitMQMessageBus()
        
        # Declare standard exchanges and queues
        for exchange_config in create_standard_exchanges():
            bus.declare_exchange(exchange_config)
        
        for queue_config in create_standard_queues():
            bus.declare_queue(queue_config)
        
        # Bind queues to exchanges
        bus.bind_queue("trade_signals", "signa_mentis.trades", "trade.signal")
        bus.bind_queue("risk_updates", "signa_mentis.risk", "risk.update")
        bus.bind_queue("news_updates", "signa_mentis.news", "news.update")
        bus.bind_queue("system_alerts", "signa_mentis.system", "system.alert")
        
        # Message handler
        def handle_message(queue_name: str, properties: pika.BasicProperties, body: bytes):
            print(f"📨 Received message on {queue_name}")
            print(f"   Properties: {properties}")
            print(f"   Body: {body.decode('utf-8')}")
        
        # Start consuming
        bus.consume_messages("trade_signals", handle_message)
        bus.consume_messages("risk_updates", handle_message)
        bus.consume_messages("news_updates", handle_message)
        bus.consume_messages("system_alerts", handle_message)
        
        # Publish test messages
        bus.publish_message("signa_mentis.trades", "trade.signal", {"action": "buy", "symbol": "XAUUSD"})
        bus.publish_message("signa_mentis.risk", "risk.update", {"risk_level": "high", "action": "reduce_position"})
        
        # Start consuming (this will block)
        print("🚀 Starting message consumption... Press Ctrl+C to stop")
        try:
            bus.start_consuming()
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        finally:
            bus.close()
    
    # Run example
    main()
