--
-- PostgreSQL database dump
--

-- Dumped from database version 17.2
-- Dumped by pg_dump version 17.2

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Data for Name: securities; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (18, 'SOXL', 'Direxion Daily Semiconductor Bull 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (19, 'TSLA', 'Tesla, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (20, 'SPXL', 'Direxion Daily S&P500 Bull 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (21, 'WGMI', 'Valkyrie Bitcoin Miners ETF', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (22, 'LABD', 'Direxion Daily S&P Biotech Bear 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (23, 'SPXS', 'Direxion Daily S&P 500 Bear 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (24, 'AAPL', 'Apple Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (25, 'AMZN', 'Amazon.com, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (26, 'BMY', 'Bristol-Myers Squibb Company', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (27, 'GOOGL', 'Alphabet Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (28, 'MELI', 'MercadoLibre, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (29, 'META', 'Meta Platforms, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (30, 'NFLX', 'Netflix, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (31, 'NVO', 'Novo Nordisk A/S', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (32, 'SQ', 'Block, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (33, 'ARGT', 'Global X MSCI Argentina ETF', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (34, 'GGAL', 'Grupo Financiero Galicia S.A.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (36, 'TNA', 'Direxion Daily Small Cap Bull 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (37, 'TZA', 'Direxion Daily Small Cap Bear 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (38, 'ACLS', 'Axcelis Technologies, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (39, 'IOVA', 'Iovance Biotherapeutics, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (40, 'CRUS', 'Cirrus Logic, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (41, 'CRM', 'Salesforce, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (7, 'NVDA', 'NVIDIA Corporation', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (8, 'TQQQ', 'ProShares UltraPro QQQ', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (9, 'SQQQ', 'ProShares UltraPro Short QQQ', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (13, 'SOXS', 'Direxion Daily Semiconductor Bear 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (14, 'LABU', 'Direxion Daily S&P Biotech Bull 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (16, 'TECL', 'Direxion Daily Technology Bull 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (17, 'MSFT', 'Microsoft Corporation', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (1, 'SPX', 'S&P 500 Index', true);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (2, 'NDQ', 'NASDAQ 100 Index', true);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (42, 'DV', 'DoubleVerify Holdings, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (43, 'TEAM', 'Atlassian Corporation', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (44, 'YOU', 'Clear Secure, Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (5, 'SPY', 'SPDR S&P 500 ETF Trust', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (10, 'QQQ', 'Invesco QQQ Trust', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (11, 'XBI', 'SPDR S&P Biotech ETF', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (12, 'SOXX', 'iShares Semiconductor ETF', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (35, 'IWM', 'iShares Russell 2000 ETF', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (45, 'ADBE', 'Adobe Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (46, 'BABA', 'Alibaba Group Holding Limited', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (47, 'PRA', 'ProAssurance Corporation', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (48, 'IBIT', 'iShares Bitcoin Trust ETF', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (49, 'AVGO', 'Broadcom Inc.', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (50, 'ORCL', 'Oracle Corporation', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (51, 'GLD', 'SPDR Gold Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (52, 'VGK', 'Vanguard FTSE Europe ETF', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (53, 'BTC-USD', 'Bitcoin USD', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (54, 'GOLD', 'Gold USD', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (55, 'EEM', 'iShares MSCI Emerging Markets ETF', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (56, 'EDC', 'Direxion Daily MSCI Emerging Markets Bull 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (57, 'EDZ', 'Direxion Daily MSCI Emerging Markets Bear 3X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (58, 'GDX', 'VanEck Gold Miners ETF', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (59, 'NUGT', 'Direxion Daily Gold Miners Index Bull 2X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (60, 'DUST', 'Direxion Daily Gold Miners Index Bear 2X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (61, 'AIBU', 'Direxion Daily AI And Big Data Bull 2X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (62, 'AIBD', 'Direxion Daily AI And Big Data Bear 2X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (63, 'UBOT', 'Direxion Daily Robotics, Artificial Intelligence & Automation Index Bull 2X Shares', false);
INSERT INTO public.securities (id, symbol, description, store_locally) OVERRIDING SYSTEM VALUE VALUES (64, 'ARTY', 'iShares Future AI & Tech ETF', false);


--
-- Data for Name: strategies; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.strategies (id, name, acronym, pool, unit_label) VALUES (2, 'Roller Coaster over RSI(14)', 'RSI(14) RC', 'RSI', '');
INSERT INTO public.strategies (id, name, acronym, pool, unit_label) VALUES (4, 'Two Bands over RSI(14)', 'RSI(14) 2B', 'RSI', '');
INSERT INTO public.strategies (id, name, acronym, pool, unit_label) VALUES (1, 'Simple Moving Average', 'SMA', 'MA', '$');
INSERT INTO public.strategies (id, name, acronym, pool, unit_label) VALUES (3, 'Simple Moving Average over RSI(14)', 'RSI(14) SMA', 'RSI', '');


--
-- Data for Name: synonyms; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.synonyms (id, provider_id, security_id, ticker) OVERRIDING SYSTEM VALUE VALUES (1, 1, 1, '^GSPC');
INSERT INTO public.synonyms (id, provider_id, security_id, ticker) OVERRIDING SYSTEM VALUE VALUES (2, 1, 2, '^NDX');
INSERT INTO public.synonyms (id, provider_id, security_id, ticker) OVERRIDING SYSTEM VALUE VALUES (3, 1, 54, 'GC=F');


--
-- Name: securities_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.securities_id_seq', 64, true);


--
-- Name: strategies_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.strategies_id_seq', 5, false);


--
-- Name: synonyms_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.synonyms_id_seq', 3, true);


--
-- PostgreSQL database dump complete
--

