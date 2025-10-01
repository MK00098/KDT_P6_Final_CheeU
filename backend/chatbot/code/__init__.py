#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU RAG+LLM Pipeline - Production Package
개인화된 멘탈헬스 AI CheeU 캡슐 생성 시스템
"""

__version__ = "1.0.0"
__author__ = "CheeU Team"
__description__ = "Evidence-based Mental Health AI with Personalized CheeU Capsules"

# Core modules
from .pipeline import CheeURagPipeline
from .vectordb import CheeUVectorDB
from .chatbot import CheeUChatbot

# Data models
from .models import (
    StressType,
    UserProfile,
    CheeUCapsule,
    TherapyMethod,
    StressTypeProfile
)

# Quick access functions
from .api import (
    create_pipeline,
    quick_healing_message,
    batch_healing_messages,
    get_stress_type_info,
    health_check,
    get_system_info,
    get_available_occupations,
    get_occupation_keywords
)

__all__ = [
    # Core classes
    "CheeURagPipeline",
    "CheeUVectorDB", 
    "CheeUChatbot",
    
    # Data models
    "StressType",
    "UserProfile",
    "CheeUCapsule",
    "TherapyMethod",
    "StressTypeProfile",
    
    # API functions
    "create_pipeline",
    "quick_healing_message",
    "batch_healing_messages",
    "get_stress_type_info",
    "health_check",
    "get_system_info",
    "get_available_occupations",
    "get_occupation_keywords"
]