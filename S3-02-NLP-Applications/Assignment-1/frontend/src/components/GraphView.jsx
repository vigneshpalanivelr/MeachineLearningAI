import React, { useEffect, useRef } from "react";
import { Network } from "vis-network/standalone";

export default function GraphView({ graphData }) {
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !graphData) return;

    // Destroy existing network instance to prevent duplicate errors
    if (networkRef.current) {
      networkRef.current.destroy();
      networkRef.current = null;
    }

    // Ensure unique nodes (remove any potential duplicates)
    const uniqueNodesMap = new Map();
    graphData.nodes.forEach((node) => {
      if (!uniqueNodesMap.has(node.id)) {
        uniqueNodesMap.set(node.id, node);
      }
    });
    const uniqueNodes = Array.from(uniqueNodesMap.values());

    const data = {
      nodes: uniqueNodes,
      edges: graphData.edges || [],
    };

    const options = {
      layout: {
        improvedLayout: true,
      },
      nodes: {
        shape: "dot",
        size: 14,
        color: {
          border: "#1976d2",
          background: "#64b5f6",
          highlight: {
            border: "#1565c0",
            background: "#42a5f5",
          },
        },
        font: {
          color: "#000",
          size: 12,
        },
      },
      edges: {
        arrows: {
          to: {
            enabled: true,
            scaleFactor: 0.5,
          },
        },
        color: {
          color: "#848484",
          highlight: "#4caf50",
        },
        font: {
          align: "middle",
          size: 10,
        },
        smooth: {
          type: "continuous",
        },
      },
      physics: {
        enabled: true,
        stabilization: {
          enabled: true,
          iterations: 200,
        },
        barnesHut: {
          gravitationalConstant: -8000,
          centralGravity: 0.3,
          springLength: 150,
          springConstant: 0.04,
        },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true,
      },
    };

    // Create new network instance
    try {
      networkRef.current = new Network(containerRef.current, data, options);
      console.log(`Graph rendered: ${uniqueNodes.length} nodes, ${data.edges.length} edges`);
    } catch (error) {
      console.error("Error creating network:", error);
    }

    // Cleanup on unmount
    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [graphData]);

  return (
    <div
      ref={containerRef}
      style={{
        height: "650px",
        background: "#fff",
        borderRadius: "8px",
        border: "1px solid #e0e0e0",
      }}
    />
  );
}
