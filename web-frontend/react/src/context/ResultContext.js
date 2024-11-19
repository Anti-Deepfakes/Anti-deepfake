import React, { createContext, useReducer, useContext } from 'react';

// 초기 상태 정의
const initialState = {
  loading: false,
  result: null,
  error: null,
};

// 리듀서 함수 정의
function resultReducer(state, action) {
  switch (action.type) {
    case 'REQUEST_START':
      return { ...state, loading: true, result: null, error: null };
    case 'REQUEST_SUCCESS':
      return { ...state, loading: false, result: action.payload, error: null };
    case 'REQUEST_ERROR':
      return { ...state, loading: false, result: null, error: action.payload };
    default:
      return state;
  }
}

// Context 생성
const ResultContext = createContext();

// Context Provider 컴포넌트
export function ResultProvider({ children }) {
  const [state, dispatch] = useReducer(resultReducer, initialState);

  return (
    <ResultContext.Provider value={{ state, dispatch }}>
      {children}
    </ResultContext.Provider>
  );
}

// Context를 사용하는 커스텀 훅
export function useResult() {
  return useContext(ResultContext);
}
