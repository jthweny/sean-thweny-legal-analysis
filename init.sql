-- Legal Analysis System Database Schema
-- PostgreSQL initialization script

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- BASELINE KNOWLEDGE REPOSITORY
-- ============================================================================

-- Baseline documents (REFINED_COMPREHENSIVE_LEGAL_ANALYSIS reference)
CREATE TABLE baseline_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    raw_content TEXT NOT NULL,
    processed_content TEXT NOT NULL,
    word_count INTEGER,
    legal_concepts JSONB,
    entities JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for baseline documents
CREATE INDEX idx_baseline_documents_hash ON baseline_documents(content_hash);
CREATE INDEX idx_baseline_documents_created ON baseline_documents(created_at);
CREATE INDEX idx_baseline_legal_concepts ON baseline_documents USING GIN(legal_concepts);
CREATE INDEX idx_baseline_entities ON baseline_documents USING GIN(entities);

-- ============================================================================
-- CASE DOCUMENTS AND PROCESSING
-- ============================================================================

-- Case documents with smart processing
CREATE TABLE case_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    file_path VARCHAR(500),
    raw_content TEXT,
    processed_content TEXT,
    word_count INTEGER,
    relevancy_score INTEGER CHECK (relevancy_score >= 0 AND relevancy_score <= 100),
    processing_status VARCHAR(20) DEFAULT 'queued' CHECK (processing_status IN ('queued', 'processing', 'completed', 'failed', 'archived')),
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for case documents
CREATE INDEX idx_case_documents_hash ON case_documents(content_hash);
CREATE INDEX idx_case_documents_status ON case_documents(processing_status);
CREATE INDEX idx_case_documents_relevancy ON case_documents(relevancy_score);
CREATE INDEX idx_case_documents_created ON case_documents(created_at);
CREATE INDEX idx_case_documents_filename ON case_documents(filename);
CREATE INDEX idx_case_documents_metadata ON case_documents USING GIN(metadata);
CREATE INDEX idx_case_documents_tags ON case_documents USING GIN(tags);

-- Document analysis results
CREATE TABLE document_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES case_documents(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    processing_time DECIMAL(10,3), -- seconds
    status VARCHAR(20) NOT NULL DEFAULT 'processing' CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    result JSONB,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd DECIMAL(10,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for document analyses
CREATE INDEX idx_document_analyses_document ON document_analyses(document_id);
CREATE INDEX idx_document_analyses_type ON document_analyses(analysis_type);
CREATE INDEX idx_document_analyses_status ON document_analyses(status);
CREATE INDEX idx_document_analyses_created ON document_analyses(created_at);
CREATE INDEX idx_document_analyses_model ON document_analyses(model_name);

-- ============================================================================
-- PERSISTENT KNOWLEDGE GRAPH
-- ============================================================================

-- Knowledge entities extracted from documents
CREATE TABLE knowledge_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    canonical_name VARCHAR(255),
    entity_type VARCHAR(50) NOT NULL, -- person, organization, concept, date, amount, etc.
    description TEXT,
    confidence_score DECIMAL(3,2) DEFAULT 0.50 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    mention_count INTEGER DEFAULT 1,
    document_count INTEGER DEFAULT 1,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    attributes JSONB,
    aliases JSONB, -- Alternative names/spellings
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for knowledge entities
CREATE INDEX idx_knowledge_entities_name ON knowledge_entities(name);
CREATE INDEX idx_knowledge_entities_canonical ON knowledge_entities(canonical_name);
CREATE INDEX idx_knowledge_entities_type ON knowledge_entities(entity_type);
CREATE INDEX idx_knowledge_entities_confidence ON knowledge_entities(confidence_score);
CREATE INDEX idx_knowledge_entities_active ON knowledge_entities(is_active);
CREATE INDEX idx_knowledge_entities_name_type ON knowledge_entities(name, entity_type);
CREATE INDEX idx_knowledge_entities_attributes ON knowledge_entities USING GIN(attributes);

-- Relationships between entities
CREATE TABLE knowledge_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    description TEXT,
    confidence_score DECIMAL(3,2) DEFAULT 0.50 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    evidence_count INTEGER DEFAULT 1,
    supporting_documents JSONB, -- Array of document IDs as evidence
    first_observed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_observed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    properties JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for knowledge relationships
CREATE INDEX idx_knowledge_relationships_source ON knowledge_relationships(source_entity_id);
CREATE INDEX idx_knowledge_relationships_target ON knowledge_relationships(target_entity_id);
CREATE INDEX idx_knowledge_relationships_type ON knowledge_relationships(relationship_type);
CREATE INDEX idx_knowledge_relationships_confidence ON knowledge_relationships(confidence_score);
CREATE INDEX idx_knowledge_relationships_active ON knowledge_relationships(is_active);
CREATE INDEX idx_knowledge_relationships_supporting ON knowledge_relationships USING GIN(supporting_documents);

-- ============================================================================
-- ACCUMULATED INSIGHTS AND PATTERNS
-- ============================================================================

-- Accumulated insights with evolution tracking
CREATE TABLE accumulated_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    insight_type VARCHAR(50) NOT NULL, -- pattern, trend, anomaly, relationship, summary, etc.
    content TEXT NOT NULL,
    summary TEXT,
    confidence_score DECIMAL(3,2) DEFAULT 0.50 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    relevance_score DECIMAL(3,2) CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0),
    novelty_score DECIMAL(3,2) CHECK (novelty_score >= 0.0 AND novelty_score <= 1.0),
    supporting_evidence JSONB,
    supporting_document_ids JSONB, -- Array of document UUIDs
    contradicts_insights UUID[], -- Array of conflicting insight IDs
    categories JSONB,
    tags JSONB,
    related_entities JSONB, -- Entity IDs this insight relates to
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    view_count INTEGER DEFAULT 0,
    validation_score DECIMAL(3,2), -- Human validation feedback
    valid_from TIMESTAMP WITH TIME ZONE,
    valid_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_validated TIMESTAMP WITH TIME ZONE
);

-- Create indexes for accumulated insights
CREATE INDEX idx_accumulated_insights_type ON accumulated_insights(insight_type);
CREATE INDEX idx_accumulated_insights_confidence ON accumulated_insights(confidence_score);
CREATE INDEX idx_accumulated_insights_active ON accumulated_insights(is_active);
CREATE INDEX idx_accumulated_insights_created ON accumulated_insights(created_at);
CREATE INDEX idx_accumulated_insights_title ON accumulated_insights(title);
CREATE INDEX idx_accumulated_insights_categories ON accumulated_insights USING GIN(categories);
CREATE INDEX idx_accumulated_insights_supporting_docs ON accumulated_insights USING GIN(supporting_document_ids);

-- Patterns discovered across multiple documents
CREATE TABLE discovered_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    description TEXT,
    pattern_definition JSONB NOT NULL, -- Formal pattern definition
    occurrence_count INTEGER DEFAULT 1,
    document_count INTEGER DEFAULT 1,
    confidence_score DECIMAL(3,2) DEFAULT 0.50 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    first_detected TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_detected TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    examples JSONB, -- Example instances of the pattern
    is_validated BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for discovered patterns
CREATE INDEX idx_discovered_patterns_type ON discovered_patterns(pattern_type);
CREATE INDEX idx_discovered_patterns_confidence ON discovered_patterns(confidence_score);
CREATE INDEX idx_discovered_patterns_active ON discovered_patterns(is_active);
CREATE INDEX idx_discovered_patterns_validated ON discovered_patterns(is_validated);

-- ============================================================================
-- BACKGROUND TASKS AND PROCESSING
-- ============================================================================

-- Background task tracking
CREATE TABLE background_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) NOT NULL UNIQUE, -- Celery task ID
    name VARCHAR(255) NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    function_name VARCHAR(255) NOT NULL,
    arguments JSONB,
    kwargs JSONB,
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    scheduled_at TIMESTAMP WITH TIME ZONE,
    max_retries INTEGER DEFAULT 3,
    retry_delay INTEGER, -- seconds
    timeout INTEGER, -- seconds
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled', 'retry', 'timeout')),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    result JSONB,
    error_message TEXT,
    error_traceback TEXT,
    progress_percent DECIMAL(5,2) CHECK (progress_percent >= 0.0 AND progress_percent <= 100.0),
    progress_message VARCHAR(500),
    retry_count INTEGER DEFAULT 0,
    next_retry_at TIMESTAMP WITH TIME ZONE,
    execution_time DECIMAL(10,3), -- seconds
    memory_usage DECIMAL(10,2), -- MB
    cpu_time DECIMAL(10,3), -- seconds
    depends_on JSONB, -- List of task IDs this task depends on
    document_id UUID REFERENCES case_documents(id),
    analysis_id UUID REFERENCES document_analyses(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for background tasks
CREATE INDEX idx_background_tasks_task_id ON background_tasks(task_id);
CREATE INDEX idx_background_tasks_status ON background_tasks(status);
CREATE INDEX idx_background_tasks_type ON background_tasks(task_type);
CREATE INDEX idx_background_tasks_priority ON background_tasks(priority);
CREATE INDEX idx_background_tasks_created ON background_tasks(created_at);
CREATE INDEX idx_background_tasks_document ON background_tasks(document_id);

-- ============================================================================
-- RESEARCH AND EXTERNAL DATA
-- ============================================================================

-- External research cache (Firecrawl results, etc.)
CREATE TABLE research_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL UNIQUE,
    query TEXT NOT NULL,
    source VARCHAR(50) NOT NULL, -- 'firecrawl', 'web_search', etc.
    result JSONB NOT NULL,
    confidence_score DECIMAL(3,2),
    cost_usd DECIMAL(10,4),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for research cache
CREATE INDEX idx_research_cache_hash ON research_cache(query_hash);
CREATE INDEX idx_research_cache_source ON research_cache(source);
CREATE INDEX idx_research_cache_expires ON research_cache(expires_at);

-- ============================================================================
-- KNOWLEDGE EVOLUTION TRACKING
-- ============================================================================

-- Track how knowledge evolves over time
CREATE TABLE knowledge_evolution (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    change_type VARCHAR(50) NOT NULL, -- 'created', 'updated', 'merged', 'deprecated'
    entity_type VARCHAR(50) NOT NULL, -- 'insight', 'entity', 'relationship', 'pattern'
    entity_id UUID NOT NULL,
    previous_state JSONB,
    new_state JSONB,
    change_description TEXT,
    triggered_by VARCHAR(100), -- What triggered the change
    confidence_change DECIMAL(5,2), -- Change in confidence score
    impact_score DECIMAL(3,2), -- How significant was this change
    affected_entities JSONB, -- Other entities affected by this change
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for knowledge evolution
CREATE INDEX idx_knowledge_evolution_type ON knowledge_evolution(change_type);
CREATE INDEX idx_knowledge_evolution_entity_type ON knowledge_evolution(entity_type);
CREATE INDEX idx_knowledge_evolution_entity_id ON knowledge_evolution(entity_id);
CREATE INDEX idx_knowledge_evolution_created ON knowledge_evolution(created_at);

-- ============================================================================
-- SYSTEM CONFIGURATION AND METADATA
-- ============================================================================

-- System configuration
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(100) NOT NULL UNIQUE,
    value JSONB NOT NULL,
    description TEXT,
    is_secret BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default configuration
INSERT INTO system_config (key, value, description) VALUES
('baseline_loaded', 'false', 'Whether baseline document has been loaded'),
('system_started_at', to_jsonb(NOW()), 'When the system was first started'),
('analysis_models', '["gemini-pro", "claude-3-sonnet", "gpt-4"]', 'Available analysis models'),
('relevancy_threshold', '70', 'Threshold for triggering deep analysis'),
('max_concurrent_analyses', '5', 'Maximum concurrent analysis tasks');

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_baseline_documents_updated_at BEFORE UPDATE ON baseline_documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_case_documents_updated_at BEFORE UPDATE ON case_documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_document_analyses_updated_at BEFORE UPDATE ON document_analyses FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_accumulated_insights_updated_at BEFORE UPDATE ON accumulated_insights FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_discovered_patterns_updated_at BEFORE UPDATE ON discovered_patterns FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_background_tasks_updated_at BEFORE UPDATE ON background_tasks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Active knowledge entities with stats
CREATE VIEW active_knowledge_entities AS
SELECT 
    e.*,
    (SELECT COUNT(*) FROM knowledge_relationships WHERE source_entity_id = e.id OR target_entity_id = e.id) as relationship_count
FROM knowledge_entities e
WHERE e.is_active = true;

-- Document processing summary
CREATE VIEW document_processing_summary AS
SELECT 
    processing_status,
    COUNT(*) as document_count,
    AVG(relevancy_score) as avg_relevancy,
    SUM(word_count) as total_words
FROM case_documents 
GROUP BY processing_status;

-- Recent insights
CREATE VIEW recent_insights AS
SELECT 
    i.*,
    array_length(i.supporting_document_ids::text[]::uuid[], 1) as evidence_count
FROM accumulated_insights i
WHERE i.is_active = true
ORDER BY i.created_at DESC;

-- ============================================================================
-- PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Enable parallel query execution
SET max_parallel_workers_per_gather = 4;
SET parallel_tuple_cost = 0.1;
SET parallel_setup_cost = 1000.0;

-- Optimize for JSONB operations
SET gin_pending_list_limit = '4MB';

COMMIT;
