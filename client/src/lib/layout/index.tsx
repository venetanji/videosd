'use client';

import { Container, Flex, VStack, Box } from '@chakra-ui/react';
import type { ReactNode } from 'react';

import Footer from './Footer';
import Header from './Header';

type LayoutProps = {
  children: ReactNode;
};

const Layout = ({ children }: LayoutProps) => {
  return (
    <Container maxW="full" width="full" flex={1} p={0}>
      <Box flex={1}>{children}</Box>
      <Footer/>
    </Container>
    
  );
};

export default Layout;
