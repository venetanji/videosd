'use client';

import { Container, Flex, VStack, Box } from '@chakra-ui/react';
import { use, useState, type ReactNode } from 'react';

import Footer from './Footer';

type LayoutProps = {
  children: ReactNode;
};

const Layout = ({ children }: LayoutProps) => {
  const [isFull, setIsFull] = useState(false);
  return (
    <Flex h='100vh' direction='column' alignItems={'stretch'} alignContent={'flex-start'}>
      {children}
      <Footer />
    </Flex>
    
  );
};

export default Layout;
