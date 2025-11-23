import React, { isValidElement } from 'react';
import Code from './Code';

type CodeWrapperProps = {
  children: React.ReactNode;
};

const CodeWrapper: React.FC<CodeWrapperProps> = ({ children }) => {
  // Check if children is a code element
  if (isValidElement(children) && children.type === 'code') {
    const props = children.props as any;
    const className = props.className || '';
    const language = className.replace('lang-', '') || 'python';
    const text = props.children;

    if (typeof text === 'string') {
      return <Code language={language} text={text} />;
    }
  }

  // Fallback
  return <pre>{children}</pre>;
};

export default CodeWrapper;
