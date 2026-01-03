import React, { useState, useEffect } from 'react';

function TestAPI() {
  const [status, setStatus] = useState('Testing...');
  const [results, setResults] = useState([]);

  useEffect(() => {
    async function test() {
      const newResults = [];
      
      try {
        // Test 1: Stock List
        newResults.push({ test: 'Stock List', status: 'Loading...', type: 'loading' });
        setResults([...newResults]);
        
        const stocksResp = await fetch('http://localhost:5000/api/stocks', {
          signal: AbortSignal.timeout(5000)
        });
        const stocksData = await stocksResp.json();
        
        newResults[0] = { 
          test: 'Stock List', 
          status: `✓ ${stocksData.stocks.length} stocks`, 
          type: 'success' 
        };
        setResults([...newResults]);

        // Test 2: RELIANCE Data
        newResults.push({ test: 'RELIANCE Data', status: 'Loading... (5-10s)', type: 'loading' });
        setResults([...newResults]);
        
        const start = Date.now();
        const relianceResp = await fetch('http://localhost:5000/api/stock/RELIANCE', {
          signal: AbortSignal.timeout(15000)
        });
        const elapsed = ((Date.now() - start) / 1000).toFixed(2);
        const relianceData = await relianceResp.json();
        
        newResults[1] = { 
          test: 'RELIANCE Data', 
          status: `✓ Loaded in ${elapsed}s - ${relianceData.tradeAnalysis.tableData.length} trades`, 
          type: 'success' 
        };
        setResults([...newResults]);
        
        setStatus('✓ All tests passed!');
        
      } catch (error) {
        newResults.push({ 
          test: 'Error', 
          status: `✗ ${error.message}`, 
          type: 'error' 
        });
        setResults([...newResults]);
        setStatus('✗ Tests failed');
      }
    }
    
    test();
  }, []);

  return (
    <div style={{ padding: '40px', background: '#0a0e1a', color: '#e4e7eb', minHeight: '100vh' }}>
      <h1>API Connection Test</h1>
      <h2 style={{ color: status.includes('✓') ? '#34d399' : '#f87171' }}>{status}</h2>
      
      <div style={{ marginTop: '20px' }}>
        {results.map((result, i) => (
          <div 
            key={i} 
            style={{
              padding: '15px',
              margin: '10px 0',
              borderRadius: '8px',
              background: result.type === 'success' ? '#065f46' : 
                         result.type === 'error' ? '#7f1d1d' : '#1e3a8a'
            }}
          >
            <strong>{result.test}:</strong> {result.status}
          </div>
        ))}
      </div>
      
      <div style={{ marginTop: '30px', padding: '15px', background: '#1a1f2e', borderRadius: '8px' }}>
        <p><strong>If all tests pass but dashboard still shows "Loading...":</strong></p>
        <ol style={{ marginLeft: '20px', color: '#9ca3af' }}>
          <li>Open browser console (F12)</li>
          <li>Look for React errors in red</li>
          <li>Check Network tab for failed requests</li>
          <li>Hard refresh: Ctrl+Shift+R</li>
        </ol>
      </div>
    </div>
  );
}

export default TestAPI;
