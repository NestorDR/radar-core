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

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: daily_data; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.daily_data (
    id integer NOT NULL,
    security_id integer NOT NULL,
    date date NOT NULL,
    open numeric(13,4) NOT NULL,
    high numeric(13,4) NOT NULL,
    low numeric(13,4) NOT NULL,
    close numeric(13,4) NOT NULL,
    volume bigint NOT NULL,
    percent_change numeric(8,2)
);


ALTER TABLE public.daily_data OWNER TO postgres;

--
-- Name: TABLE daily_data; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.daily_data IS 'Daily prices (OHLC) and indicators for the securities';


--
-- Name: daily_data_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.daily_data ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.daily_data_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
    CYCLE
);


--
-- Name: ratios; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ratios (
    id integer NOT NULL,
    symbol character varying(10) NOT NULL,
    strategy_id integer NOT NULL,
    timeframe smallint NOT NULL,
    inputs character varying(50) NOT NULL,
    is_long_position boolean NOT NULL,
    is_in_process boolean DEFAULT false NOT NULL,
    from_date date NOT NULL,
    to_date date NOT NULL,
    initial_price numeric(12,2) NOT NULL,
    final_price numeric(9,2) NOT NULL,
    net_change real NOT NULL,
    signals smallint NOT NULL,
    winnings real NOT NULL,
    losses real NOT NULL,
    net_profit real NOT NULL,
    expected_value real NOT NULL,
    win_probability real NOT NULL,
    loss_probability real NOT NULL,
    average_win real NOT NULL,
    average_loss real NOT NULL,
    min_percentage_change_to_win numeric(6,2) NOT NULL,
    max_percentage_change_to_win numeric(6,2) NOT NULL,
    total_sessions smallint NOT NULL,
    winning_sessions smallint NOT NULL,
    losing_sessions smallint NOT NULL,
    percentage_exposure real NOT NULL,
    first_input_date date NOT NULL,
    last_input_date date NOT NULL,
    last_output_date date,
    last_input_price numeric(12,2),
    last_output_price numeric(12,2),
    last_stop_loss numeric(12,2)
);


ALTER TABLE public.ratios OWNER TO postgres;

--
-- Name: TABLE ratios; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.ratios IS 'ratios to evaluate the performance of speculation/investment strategies';


--
-- Name: COLUMN ratios.symbol; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.symbol IS 'Acronym identifier of financial instrument';


--
-- Name: COLUMN ratios.timeframe; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.timeframe IS 'Time frames: 1.Intraday, 2.Daily, 3.Weekly, 4.Monthly';


--
-- Name: COLUMN ratios.inputs; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.inputs IS 'Defines the independent variables of the strategy, for example for a moving average it is the length in sessions of the period for calculating the average';


--
-- Name: COLUMN ratios.is_long_position; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.is_long_position IS 'Defines whether the ratio relates to a long or short market position';


--
-- Name: COLUMN ratios.is_in_process; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.is_in_process IS 'Flag indicating whether the record can be deleted during any running process';


--
-- Name: COLUMN ratios.from_date; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.from_date IS 'Indicates the date from which the strategy was tested to identify results and ratios';


--
-- Name: COLUMN ratios.to_date; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.to_date IS 'Indicates the date up to which the strategy was tested to identify results and ratios.';


--
-- Name: COLUMN ratios.initial_price; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.initial_price IS 'Initial price of the period in which the strategy was tested';


--
-- Name: COLUMN ratios.final_price; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.final_price IS 'Final price of the period in which the strategy was tested';


--
-- Name: COLUMN ratios.net_change; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.net_change IS 'Percentage change of the final price over the initial price';


--
-- Name: COLUMN ratios.signals; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.signals IS 'Number of trade signals identified by the strategy';


--
-- Name: COLUMN ratios.winnings; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.winnings IS 'Total gain on positive/winning operations';


--
-- Name: COLUMN ratios.losses; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.losses IS 'Total loss on negative/losing operations.';


--
-- Name: COLUMN ratios.net_profit; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.net_profit IS 'Percentage of net profit got following the input and output signals. Formula: (winnings_ - losses_) / initial_price';


--
-- Name: COLUMN ratios.expected_value; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.expected_value IS 'Mathematical expectation of the strategy. Formula: (win_probability * average_win) + (loss_probability * average_loss).';


--
-- Name: COLUMN ratios.win_probability; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.win_probability IS 'Percentage of positive/winning operations';


--
-- Name: COLUMN ratios.loss_probability; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.loss_probability IS 'Percentage of negative/losing operations';


--
-- Name: COLUMN ratios.average_win; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.average_win IS 'Average profit of the positive/winning operations';


--
-- Name: COLUMN ratios.average_loss; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.average_loss IS 'Average loss of the negative/losing operations';


--
-- Name: COLUMN ratios.min_percentage_change_to_win; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.min_percentage_change_to_win IS 'Minimum percentage change of input sessions for the positive/winning operations';


--
-- Name: COLUMN ratios.max_percentage_change_to_win; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.max_percentage_change_to_win IS 'Maximum percentage change of input sessions for the positive/winning operations';


--
-- Name: COLUMN ratios.total_sessions; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.total_sessions IS 'Total number of sessions to which the strategy has been evaluated';


--
-- Name: COLUMN ratios.winning_sessions; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.winning_sessions IS 'Number of sessions spent/elapsed during positive/winning operations';


--
-- Name: COLUMN ratios.losing_sessions; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.losing_sessions IS 'Number of sessions spent/elapsed during negative/losing operations';


--
-- Name: COLUMN ratios.percentage_exposure; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.percentage_exposure IS 'Percentage time during which the strategy was active. Formula: (winning_sessions + losing_sessions) / total_sessions';


--
-- Name: COLUMN ratios.first_input_date; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.first_input_date IS 'Date of the first input signal';


--
-- Name: COLUMN ratios.last_input_date; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.last_input_date IS 'Date of the last input signal';


--
-- Name: COLUMN ratios.last_output_date; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.last_output_date IS 'Date of the last output signal';


--
-- Name: COLUMN ratios.last_input_price; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.last_input_price IS 'Input price for the last position opened by the tested strategy';


--
-- Name: COLUMN ratios.last_output_price; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.last_output_price IS 'Output price for the last position opened by the tested strategy';


--
-- Name: COLUMN ratios.last_stop_loss; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.ratios.last_stop_loss IS 'Stop loss for the last position opened by the tested strategy';


--
-- Name: ratios_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.ratios ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.ratios_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
    CYCLE
);


--
-- Name: strategies; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.strategies (
    id integer NOT NULL,
    name character varying(50) NOT NULL,
    acronym character varying(25) NOT NULL,
    pool character varying(10) DEFAULT ''::character varying NOT NULL,
    unit_label character varying(5) DEFAULT ''::character varying NOT NULL
);


ALTER TABLE public.strategies OWNER TO postgres;

--
-- Name: TABLE strategies; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.strategies IS 'Speculation/investment strategies on financial instruments';


--
-- Name: ratios_view; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.ratios_view AS
 SELECT ratios.symbol AS "Symbol",
        CASE ratios.is_long_position
            WHEN false THEN 'Short'::text
            ELSE 'Long'::text
        END AS "Position",
        CASE ratios.timeframe
            WHEN 1 THEN 'Intra'::text
            WHEN 2 THEN 'Day'::text
            WHEN 3 THEN 'Week'::text
            WHEN 4 THEN 'Month'::text
            ELSE '?'::text
        END AS "Frame",
    strategies.acronym AS "Strategy",
        CASE ratios.timeframe
            WHEN 3 THEN LEAST((ratios.last_input_date + 4), CURRENT_DATE)
            ELSE ratios.last_input_date
        END AS "Input Date",
    ratios.last_input_price AS "Input Price",
    ratios.last_stop_loss AS "Stop Loss",
    ratios.last_output_price AS "Output Price",
    ratios.last_output_date AS "Output Date",
    ratios.win_probability AS "Gain Probability",
        CASE
            WHEN ((strategies.acronym)::text ~~ '%(14)%'::text) THEN replace(replace(TRIM(BOTH '{}'::text FROM ratios.inputs), '"'::text, ''::text), 'period: 14, '::text, ''::text)
            ELSE replace(TRIM(BOTH '{}'::text FROM ratios.inputs), '"'::text, ''::text)
        END AS "Inputs",
    ratios.final_price AS "Final Price",
    ratios.initial_price AS "Initial Price",
    ratios.from_date AS "From Date",
    ratios.to_date AS "To Date",
    ratios.net_change AS "Net Change",
    ratios.net_profit AS "Net Profit",
    ratios.signals AS "Signals",
    ratios.winnings AS "Gains",
    ratios.losses AS "Losses",
    ratios.expected_value AS "Expected Value",
    ratios.loss_probability AS "Loss Probability",
    ratios.average_win AS "Average Gain",
    ratios.average_loss AS "Average Loss",
    (ratios.min_percentage_change_to_win / (100)::numeric) AS "Min % to Gain",
    (ratios.max_percentage_change_to_win / (100)::numeric) AS "MAX % to Gain",
    ratios.total_sessions AS "Total Sessions",
    ratios.winning_sessions AS "Gains Sessions",
    ratios.losing_sessions AS "Losses Sessions",
    ratios.percentage_exposure AS "% Exposure"
   FROM (public.ratios
     JOIN public.strategies ON ((strategies.id = ratios.strategy_id)))
  ORDER BY ratios.timeframe, ratios.symbol, ratios.is_long_position DESC, ratios.strategy_id;


ALTER VIEW public.ratios_view OWNER TO postgres;

--
-- Name: securities; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.securities (
    id integer NOT NULL,
    symbol character varying(10) NOT NULL,
    description character varying(100) NOT NULL,
    store_locally boolean DEFAULT false NOT NULL
);


ALTER TABLE public.securities OWNER TO postgres;

--
-- Name: TABLE securities; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.securities IS 'Marketable financial instruments';


--
-- Name: COLUMN securities.symbol; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.securities.symbol IS 'Acronym identifier of financial instrument';


--
-- Name: COLUMN securities.store_locally; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.securities.store_locally IS 'Flag indicating whether prices obtained from the cloud should be saved in the database';


--
-- Name: securities_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.securities ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.securities_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
    CYCLE
);


--
-- Name: strategies_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.strategies ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.strategies_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: synonyms; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.synonyms (
    id integer NOT NULL,
    provider_id integer NOT NULL,
    security_id integer NOT NULL,
    ticker character varying(10) NOT NULL
);


ALTER TABLE public.synonyms OWNER TO postgres;

--
-- Name: TABLE synonyms; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.synonyms IS 'Synonyms of security symbols in different quote providers.
The providerId column is managed without a master table, and its values are:
1 -> Yahoo';


--
-- Name: synonyms_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.synonyms ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.synonyms_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: daily_data dailydata_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_data
    ADD CONSTRAINT dailydata_pkey PRIMARY KEY (id);


--
-- Name: daily_data dailydata_securityid_date_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_data
    ADD CONSTRAINT dailydata_securityid_date_unique UNIQUE (security_id, date);


--
-- Name: ratios ratios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ratios
    ADD CONSTRAINT ratios_pkey PRIMARY KEY (id);


--
-- Name: ratios ratios_symbol_strategy_inputs_timeframe_islong_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ratios
    ADD CONSTRAINT ratios_symbol_strategy_inputs_timeframe_islong_unique UNIQUE (symbol, strategy_id, inputs, timeframe, is_long_position);


--
-- Name: securities securities_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.securities
    ADD CONSTRAINT securities_pkey PRIMARY KEY (id);


--
-- Name: securities securities_symbol_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.securities
    ADD CONSTRAINT securities_symbol_unique UNIQUE (symbol);


--
-- Name: strategies strategies_acronym_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.strategies
    ADD CONSTRAINT strategies_acronym_unique UNIQUE (acronym);


--
-- Name: strategies strategies_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.strategies
    ADD CONSTRAINT strategies_pkey PRIMARY KEY (id);


--
-- Name: synonyms synonyms_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.synonyms
    ADD CONSTRAINT synonyms_pkey PRIMARY KEY (id);


--
-- Name: synonyms synonyms_providerid_securityid_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.synonyms
    ADD CONSTRAINT synonyms_providerid_securityid_unique UNIQUE (provider_id, security_id);


--
-- Name: daily_data dailydata_securities_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_data
    ADD CONSTRAINT dailydata_securities_fkey FOREIGN KEY (security_id) REFERENCES public.securities(id) NOT VALID;


--
-- Name: ratios ratios_strategies_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ratios
    ADD CONSTRAINT ratios_strategies_fkey FOREIGN KEY (strategy_id) REFERENCES public.strategies(id);


--
-- Name: synonyms synonyms_securities_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.synonyms
    ADD CONSTRAINT synonyms_securities_fkey FOREIGN KEY (security_id) REFERENCES public.securities(id) NOT VALID;


--
-- Name: TABLE ratios; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.ratios TO webuser;


--
-- Name: TABLE strategies; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.strategies TO webuser;


--
-- Name: TABLE securities; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.securities TO webuser;


--
-- PostgreSQL database dump complete
--

